"""Atomically publish host compiler cache files on same-directory-rename mounts."""
import argparse
import hashlib
from pathlib import Path
ORIGINAL_SHA = '56cf65e0379184a5f207e85f0c1ae9e6ff3938eeaf9245d31d356367bceed318'
PATCHED_SHA = 'c76ffd3bd887759ce14b7076d36dfd7fd1e0befa6ef5bd8a810984ba25d2dabe'
OLD = '        # Random ID to avoid any collisions\n        rnd_id = str(uuid.uuid4())\n        # we use the PID in case a bunch of these around so we can see what PID made it\n        pid = os.getpid()\n        # use temp dir to be robust against program interruptions\n        temp_dir = os.path.join(self.cache_dir, f"tmp.pid_{pid}_{rnd_id}")\n        os.makedirs(temp_dir, exist_ok=True)\n        temp_path = os.path.join(temp_dir, filename)\n\n        mode = "wb" if binary else "w"\n        with open(temp_path, mode) as f:\n            f.write(data)\n        # Replace is guaranteed to be atomic on POSIX systems if it succeeds\n        # so filepath cannot see a partial write\n        os.replace(temp_path, filepath)\n        os.removedirs(temp_dir)\n        return filepath\n'
NEW = '        # A sibling temporary file also works on mounts that reject\n        # cross-directory rename. Keep atomic publication and unique writers.\n        descriptor, temp_path = tempfile.mkstemp(\n            prefix=f".tmp.pid_{os.getpid()}_", dir=os.path.dirname(filepath))\n        try:\n            mode = "wb" if binary else "w"\n            with os.fdopen(descriptor, mode) as stream:\n                stream.write(data)\n            # Readers see either the previous complete file or the new one.\n            os.replace(temp_path, filepath)\n        finally:\n            if os.path.exists(temp_path):\n                os.unlink(temp_path)\n        return filepath\n'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--site-packages',type=Path,required=True)
    args=parser.parse_args()
    target=args.site_packages/'triton/runtime/cache.py'
    source=target.read_bytes()
    digest=hashlib.sha256(source).hexdigest()
    if digest==PATCHED_SHA:
        print('HOST_CACHE_PUBLICATION_PATCH_ALREADY_MATCHES')
        return
    if digest!=ORIGINAL_SHA:
        raise RuntimeError('Unexpected vendor cache source hash; file unchanged')
    text=source.decode()
    if text.count(OLD)!=1:
        raise RuntimeError('Pinned publication block differs; file unchanged')
    updated=text.replace('import uuid\n','import uuid\nimport tempfile\n',1).replace(OLD,NEW)
    if hashlib.sha256(updated.encode()).hexdigest()!=PATCHED_SHA:
        raise RuntimeError('Patch identity differs; file unchanged')
    compile(updated,str(target),'exec')
    backup=target.with_suffix('.py.h3-publication-original')
    if backup.exists() and backup.read_bytes()!=source:
        raise RuntimeError('Existing backup differs; file unchanged')
    backup.write_bytes(source)
    target.write_bytes(updated.encode())
    print('HOST_CACHE_PUBLICATION_PATCH_MATCHES',PATCHED_SHA)

if __name__=='__main__':main()
