"""Check every outer archive member against SHA256SUMS.json, without extraction."""
import argparse
import hashlib
import json
import zipfile
from pathlib import PurePosixPath


def verify(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive names')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name:
                raise ValueError('Unsafe archive path')
        rows = json.loads(archive.read('SHA256SUMS.json'))
        declared = [row['file'] for row in rows]
        if len(set(declared)) != len(declared) or set(declared) != set(names)-{'SHA256SUMS.json'}:
            raise ValueError('Incomplete archive checksum manifest')
        for row in rows:
            if archive.getinfo(row['file']).file_size != row['bytes']:
                raise ValueError('Size mismatch: '+row['file'])
            with archive.open(row['file']) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                    raise ValueError('SHA mismatch: '+row['file'])
        state = json.loads(archive.read('completion-status.json'))
        return dict(status='PASS', verified_entries=len(rows), completion_status=state['status'],
                    ready_for_preliminary_submission=state['ready_for_preliminary_submission'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    print(json.dumps(verify(parser.parse_args().archive), ensure_ascii=False))
