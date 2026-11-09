"""Verify every declared release file before constructing or running anything."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def verify(manifest, read):
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported release manifest')
    for name, row in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('Unsafe declared path')
        raw = read(name)
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Release file differs: ' + name)
    print('RELEASE_FILE_HASHES_PASS', len(manifest['files']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.archive:
        with zipfile.ZipFile(args.archive) as archive:
            if archive.testzip() is not None:
                raise ValueError('Archive CRC verification failed')
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate archive members')
            for name in names:
                p = PurePosixPath(name)
                if p.is_absolute() or '..' in p.parts or '\\' in name:
                    raise ValueError('Unsafe archive path')
            candidates = [n for n in names if PurePosixPath(n).name == 'content-manifest.json']
            if len(candidates) != 1:
                raise ValueError('Expected one release manifest')
            path = candidates[0]
            prefix = path[:-len('content-manifest.json')]
            manifest = json.loads(archive.read(path))
            expected = {prefix + name for name in manifest['files']} | {path}
            if set(names) != expected:
                raise ValueError('Archive includes undeclared or missing members')
            verify(manifest, lambda name: archive.read(prefix + name))
    else:
        root = args.root.resolve()
        manifest = json.loads((root/'content-manifest.json').read_text(encoding='utf-8'))
        verify(manifest, lambda name: (root/Path(name)).read_bytes())


if __name__ == '__main__':
    main()
