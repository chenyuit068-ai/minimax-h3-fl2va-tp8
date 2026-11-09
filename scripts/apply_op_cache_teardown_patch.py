"""Apply the separately recorded host-only Core descriptor cleanup fix."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--core',type=Path,required=True);a=p.parse_args()
package=Path(__file__).resolve().parents[1]
manifest=json.loads((package/'platform/op-cache-teardown-manifest.json').read_text())
patch=package/'platform/InfiniCore-op-cache-teardown.patch'
target=a.core/'include/infinicore/ops/common/cache.hpp'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(patch)==manifest['patch_sha256'],'Platform patch changed'
if sha(target)!=manifest['after_sha256']:
    assert sha(target) in (manifest['before_sha256'],manifest['normalized_before_sha256']),'Unexpected Core cache source; do not overwrite'
    reference=target.with_name(target.name+'.h3-cache-teardown-reference')
    if reference.exists():assert sha(reference)==manifest['before_sha256']
    else:shutil.copyfile(target,reference)
    # The offline checkout retains Windows CRLF; unified diff uses LF.
    normalized=reference.read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(normalized).hexdigest()==manifest['normalized_before_sha256']
    target.write_bytes(normalized)
    git=['git','-c','core.autocrlf=false','-c','core.eol=lf','-C',str(a.core)]
    subprocess.run([*git,'apply','--check',str(patch)],check=True)
    subprocess.run([*git,'apply',str(patch)],check=True)
    target.write_bytes(target.read_bytes().replace(b'\r\n',b'\n'))
assert sha(target)==manifest['after_sha256']
print('OP_CACHE_TEARDOWN_PATCH_APPLIED',sha(target),flush=True)
