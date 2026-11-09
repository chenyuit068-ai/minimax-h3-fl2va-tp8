"""Plan or resume the pinned public modular checkpoint without model conversion."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download

REPO = 'MiniMaxAI/MiniMax-H3'
REVISION = '42ed227ee7df40d41602854ae760620d6eb651fe'
PREFIXES = ('transformer/', 'text_encoder/', 'vae/', 'audio_vae/',
            'tokenizer/', 'processor/', 'scheduler/', 'audio_scheduler/',
            'video_processor/', 'image_processor/', 'feature_extractor/')
ROOT_FILES = {'model_index.json', 'modular_model_index.json'}


def download_weight(row, destination, *, reserve_bytes=0, abort=None):
    """Verified standard HTTPS streaming, with an explicit byte-range resume."""
    path = destination/row['file']
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return path
    partial = path.with_name(path.name+'.download.part')
    expected = row['bytes']
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{row['file']}"
    last_error = None
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected:
            break
        if offset > expected:
            raise ValueError(f'Partial file exceeds official size: {partial}')
        try:
            headers = {'Range': f'bytes={offset}-{expected-1}'} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(15, 30)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    span = response.headers.get('Content-Range', '')
                    if not span.startswith(f'bytes {offset}-') or not span.endswith(f'/{expected}'):
                        raise ValueError(f'Unexpected HTTP byte range: {span}')
                elif response.status_code != 200 or offset:
                    raise ValueError('Server did not honor the required resume range')
                reported = offset
                with partial.open('ab') as target:
                    for chunk in response.iter_content(4*1024*1024):
                        if not chunk:
                            continue
                        if abort is not None and abort.is_set():
                            raise RuntimeError('Download stopped after another shard failed')
                        if shutil.disk_usage(path.parent).free < reserve_bytes+len(chunk):
                            if abort is not None:abort.set()
                            raise RuntimeError('Free disk space reached the configured reserve')
                        offset += len(chunk)
                        if offset > expected:
                            raise ValueError('Server sent more bytes than the official shard size')
                        target.write(chunk)
                        if offset-reported >= 64*1024*1024:
                            target.flush()
                            print('DOWNLOAD_BYTES', row['file'], offset, '/', expected, flush=True)
                            reported = offset
            if partial.stat().st_size != expected:
                raise OSError('HTTP response ended before the shard was complete')
            break
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            print('DOWNLOAD_RETRY', attempt+1, row['file'], type(exc).__name__, flush=True)
            time.sleep(min(2**attempt, 15))
    if not partial.exists() or partial.stat().st_size != expected:
        raise OSError(f"Unable to complete official shard {row['file']}") from last_error
    with partial.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if digest != row['sha256']:
        raise ValueError(f"Official shard SHA256 mismatch: {row['file']}")
    partial.rename(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Download the planned files; default only writes a plan')
    parser.add_argument('--reserve-gib', type=int, default=20)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.reserve_gib < 0:
        raise ValueError('reserve-gib must be nonnegative')
    if not 1 <= args.workers <= 4:
        raise ValueError('workers must be between 1 and 4')
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    info = HfApi(token=False).model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION, 'The requested model revision changed'
    files = []
    for sibling in info.siblings:
        name = sibling.rfilename
        if name not in ROOT_FILES and not name.startswith(PREFIXES):
            continue
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe repository path')
        if sibling.size is None:
            raise ValueError(f'Missing public file size: {name}')
        lfs = sibling.lfs
        digest = getattr(lfs, 'sha256', None) if lfs is not None else None
        if isinstance(lfs, dict):
            digest = lfs.get('sha256')
        if name.endswith('.safetensors') and not digest:
            raise ValueError(f'Missing weight integrity hash: {name}')
        files.append(dict(file=name, bytes=sibling.size, sha256=digest))
    assert all(any(r['file'].startswith(prefix) and r['file'].endswith('.safetensors')
                   for r in files) for prefix in PREFIXES[:4])
    total = sum(r['bytes'] for r in files)
    plan = dict(repo=REPO, revision=REVISION, files=files, total_bytes=total,
                weight_bytes=sum(r['bytes'] for r in files if r['file'].endswith('.safetensors')),
                complete=False, full_fl2va_tested=False)
    manifest = dest/'asset-download-manifest.json'
    manifest.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='PLAN_ONLY', files=len(files), bytes=total,
                         free_bytes=shutil.disk_usage(dest).free)), flush=True)
    if not args.execute:
        return
    def remaining(row):
        path=dest/row['file']
        if path.is_file() and path.stat().st_size==row['bytes']:
            return 0
        partial=path.with_name(path.name+'.download.part')
        completed=partial.stat().st_size if partial.exists() else 0
        if completed>row['bytes']:
            raise ValueError('Partial file exceeds its official size')
        return row['bytes']-completed
    # A component can live on another volume through a directory junction.
    volumes={}
    for row in files:
        parent=(dest/row['file']).parent
        parent.mkdir(parents=True,exist_ok=True)
        resolved=parent.resolve();volume=resolved.stat().st_dev
        if volume not in volumes:volumes[volume]=[resolved,0]
        volumes[volume][1]+=remaining(row)
    reserve_bytes=args.reserve_gib*1024**3
    for directory,needed in volumes.values():
        if shutil.disk_usage(directory).free < needed+reserve_bytes:
            raise OSError(f'Insufficient free space for checkpoint plus reserve on {directory.anchor}')
    abort=threading.Event()
    def verify(row):
        path = (download_weight(row, dest,reserve_bytes=reserve_bytes,abort=abort) if row['file'].endswith('.safetensors') else
                Path(hf_hub_download(REPO, row['file'], revision=REVISION, local_dir=dest, token=False)))
        if path.stat().st_size != row['bytes']:
            raise ValueError(f"Downloaded size mismatch: {row['file']}")
        actual = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        if row['sha256'] is not None and actual != row['sha256']:
            raise ValueError(f"Downloaded hash mismatch: {row['file']}")
        return actual

    def record(row, digest):
        row['verified_sha256'] = digest
        manifest.write_text(json.dumps(plan, indent=2), encoding='utf-8')
        print('VERIFIED', row['file'], flush=True)

    for row in files:
        if not row['file'].endswith('.safetensors'):
            record(row, verify(row))
    errors=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(verify,row):row for row in files if row['file'].endswith('.safetensors')}
        for future in as_completed(pending):
            row=pending[future]
            try:
                record(row,future.result())
            except Exception as exc:
                abort.set()
                errors.append(dict(file=row['file'],error=type(exc).__name__,message=str(exc)))
                print('SHARD_FAILED',row['file'],type(exc).__name__,flush=True)
    if errors:
        plan['errors']=errors
        manifest.write_text(json.dumps(plan,indent=2),encoding='utf-8')
        raise RuntimeError(f'{len(errors)} checkpoint files failed; retained partial files for resumption')
    index = json.loads((dest/'model_index.json').read_text())
    if index.get('_class_name') != 'MiniMaxH3ModularPipeline':
        raise ValueError('The pinned root does not match the native modular entry')
    plan['complete'] = True
    manifest.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print('PUBLIC_ASSETS_DOWNLOADED_AND_HASH_VERIFIED; MODEL_NOT_TESTED', flush=True)


if __name__ == '__main__':
    main()
