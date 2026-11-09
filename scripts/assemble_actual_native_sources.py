"""Validate actual-build source evidence and reconstruct a fresh source tree.

This is a host file operation only. It never installs dependencies, compiles,
requests accelerators, or modifies an existing source checkout.
"""
import argparse
import base64
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import zlib


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_payload(manifest_path):
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    parts=[]
    for row in manifest['parts']:
        path=manifest_path.parent/row['file']
        raw=path.read_bytes()
        if len(raw)!=row['bytes'] or digest(raw)!=row['sha256']:
            raise ValueError('Source evidence part differs: '+row['file'])
        parts.append(raw)
    compressed=base64.b64decode(b''.join(parts),validate=True)
    if len(compressed)!=manifest['compressed_bytes']:
        raise ValueError('Compressed source evidence size differs')
    payload=zlib.decompress(compressed)
    if len(payload)!=manifest['payload_bytes'] or digest(payload)!=manifest['payload_sha256']:
        raise ValueError('Decoded source evidence differs')
    bundle=json.loads(payload)
    if len(bundle['files'])!=manifest['source_files']:
        raise ValueError('Source evidence file count differs')
    if bundle['contains_weights'] or bundle['contains_device_binaries']:
        raise ValueError('Expected source-only evidence')
    report=base64.b64decode(bundle['build_report_base64'],validate=True)
    if digest(report)!=manifest['build_report_sha256']:
        raise ValueError('Actual native build report differs')
    for name,row in bundle['files'].items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[:2] not in (
                ('Projects','InfiniCore'),('Projects','InfiniLM')):
            raise ValueError('Unexpected source overlay path: '+name)
        raw=base64.b64decode(row['data_base64'],validate=True)
        if len(raw)!=row['bytes'] or digest(raw)!=row['sha256']:
            raise ValueError('Source overlay content differs: '+name)
        raw.decode('utf-8')
    return manifest,bundle,payload,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--source-archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output=a.output.resolve()
    if os.name=='nt' and not str(a.output).startswith('\\\\?\\'):
        a.output=Path('\\\\?\\'+str(a.output))
    manifest,bundle,payload,report=checked_payload(a.manifest)
    if digest(a.source_archive.read_bytes())!=manifest['source_archive_sha256']:
        raise ValueError('Pinned base source archive differs')
    if a.output.exists():raise FileExistsError('Choose a new output directory')
    # Validate all member paths before creating a new destination.
    with tarfile.open(a.source_archive) as archive:
        members=archive.getmembers()
        for member in members:
            name=PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or name.parts[0]!='Projects' or not (member.isfile() or member.isdir()):
                raise ValueError('Unexpected base source archive member: '+member.name)
        a.output.mkdir(parents=True,exist_ok=False)
        source=a.output/'source';source.mkdir()
        # tarfile mixes POSIX member separators with the Windows extended path
        # prefix when restoring directory attributes. Normalize every member
        # through Path instead; all members were checked above, and the new
        # destination contains no pre-existing links.
        for member in members:
            target=source.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as src,target.open('wb') as dst:
                    shutil.copyfileobj(src,dst)
                if os.name!='nt':target.chmod(member.mode&0o777)
    patches={'InfiniCore':[],'InfiniLM':[]}
    records=[]
    for name,row in sorted(bundle['files'].items()):
        path=source/Path(name)
        before=path.read_bytes() if path.exists() else b''
        after=base64.b64decode(row['data_base64'],validate=True)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(after)
        project=PurePosixPath(name).parts[1]
        relative=PurePosixPath(*PurePosixPath(name).parts[2:]).as_posix()
        patches[project].extend(difflib.unified_diff(
            before.decode('utf-8').splitlines(True),after.decode('utf-8').splitlines(True),
            fromfile='a/'+relative if before else '/dev/null',tofile='b/'+relative))
        records.append(dict(file=name,base_sha256=digest(before) if before else None,
                            actual_build_sha256=digest(after),bytes=len(after)))
    for name,lines in patches.items():
        (a.output/(name+'-actual-build.patch')).write_text(''.join(lines),encoding='utf-8',newline='\n')
    (a.output/'native-source-overlays-r10.json').write_bytes(payload)
    (a.output/'native-build-r10.json').write_bytes(report)
    result=dict(status='ACTUAL_NATIVE_SOURCE_RECONSTRUCTION_PASS_NOT_A_REBUILD',
        source_archive_sha256=manifest['source_archive_sha256'],payload_sha256=digest(payload),
        build_report_sha256=digest(report),source_files=records,
        compilation_executed=False,gpu_operations_executed=False)
    (a.output/'source-reconstruction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],source_files=len(records),output=str(a.output)),indent=2))


if __name__=='__main__':main()
