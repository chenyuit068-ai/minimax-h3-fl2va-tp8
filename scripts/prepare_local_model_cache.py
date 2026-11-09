"""Stage byte-identical original weights on compute-local storage, never results."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time


def cache(source, output, expected, components=('transformer',), reserve_bytes=64*1024**3):
    source = Path(source).resolve(); output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Use a fresh local cache directory')
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Cache and original model must be independent directories')
    if not output.parent.is_dir(): raise FileNotFoundError('Cache parent must exist')
    components = set(components)
    rows = expected['files']
    if len({r['file'] for r in rows}) != len(rows): raise ValueError('Duplicate manifest paths')
    available_components = set()
    needed = 0
    for row in rows:
        rel = PurePosixPath(row['file'])
        if rel.is_absolute() or '..' in rel.parts or '\\' in row['file'] or not rel.parts:
            raise ValueError('Unsafe asset path')
        available_components.add(rel.parts[0])
        p = source/rel
        if not p.is_file() or p.stat().st_size != row['bytes']:
            raise ValueError('Missing or wrong-sized original asset: '+row['file'])
        if rel.parts[0] in components: needed += row['bytes']
    if not components <= available_components: raise ValueError('Unknown cache component')
    if reserve_bytes < 0 or shutil.disk_usage(output.parent).free < needed+reserve_bytes:
        raise ValueError('Insufficient local storage with reserved headroom')
    output.mkdir()
    started = time.monotonic(); verified = []; linked = set()
    for row in rows:
        rel = PurePosixPath(row['file']); src = source/rel
        digest = hashlib.sha256()
        cached = rel.parts[0] in components
        if cached:
            dest = output/rel; dest.parent.mkdir(parents=True, exist_ok=True)
            partial = dest.with_name(dest.name+'.copying')
            with src.open('rb') as incoming, partial.open('xb') as outgoing:
                while block := incoming.read(8*1024**2):
                    digest.update(block); outgoing.write(block)
                outgoing.flush(); os.fsync(outgoing.fileno())
            if digest.hexdigest() != row['sha256']:
                raise ValueError('Original SHA256 differs: '+row['file'])
            with partial.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                    raise ValueError('Local cache readback SHA256 differs: '+row['file'])
            os.replace(partial, dest)
        else:
            with src.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256')
            if digest.hexdigest() != row['sha256']:
                raise ValueError('Original SHA256 differs: '+row['file'])
            if rel.parts[0] not in linked:
                top = source/rel.parts[0]
                os.symlink(top, output/rel.parts[0], target_is_directory=top.is_dir())
                linked.add(rel.parts[0])
        verified.append(dict(row, verified_sha256=digest.hexdigest()))
        print(json.dumps(dict(event='ORIGINAL_ASSET_VERIFIED', file=row['file'], locally_cached=cached)), flush=True)
    manifest = dict(repo=expected['repo'], revision=expected['revision'], complete=True,
        files=verified, source_directory=str(source),
        verification='Full original SHA256; copied bytes additionally verified by local readback',
        cached_components=sorted(components), cached_bytes=needed,
        weight_values_modified=False, generated_results_cached=False,
        prepared_at=datetime.datetime.now().astimezone().isoformat(),
        wall_seconds=time.monotonic()-started)
    (output/'asset-download-manifest.json').write_text(json.dumps(manifest, indent=2))
    return dict(status='ORIGINAL_MODEL_LOCAL_CACHE_READY', model_path=str(output),
                cached_bytes=needed, verified_files=len(rows), source_files_modified=False,
                generated_results_cached=False, wall_seconds=manifest['wall_seconds'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--components', default='transformer')
    p.add_argument('--reserve-gib', type=int, default=64)
    a = p.parse_args()
    expected = json.loads(a.manifest.read_text())
    assert expected['repo']=='MiniMaxAI/MiniMax-H3' and expected['revision']=='42ed227ee7df40d41602854ae760620d6eb651fe'
    print(json.dumps(cache(a.model_path,a.output,expected,a.components.split(','),a.reserve_gib*1024**3)), flush=True)
