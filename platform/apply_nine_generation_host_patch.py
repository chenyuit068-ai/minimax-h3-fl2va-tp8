"""Apply the hash-pinned Nine host generation option; no device code is added."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site-packages',type=Path,required=True);a=p.parse_args()
here=Path(__file__).resolve().parent
manifest=json.loads((here/'compiler-reproduction-manifest.json').read_text())
root=a.site_packages.resolve();target=root/'ninetoothed/generation.py'
patch=here/'nine-generation-host.patch'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(patch)==manifest['patches'][patch.name]
after=manifest['qualified_compiler_sources']['nine_generation']
if sha(target)!=after:
    assert sha(target)==manifest['nine_generation_before_sha256'],'Unexpected Nine source; refusing overwrite'
    backup=target.with_name(target.name+'.h3-original')
    if backup.exists():assert sha(backup)==sha(target)
    else:shutil.copyfile(target,backup)
    git=['git','-c','core.autocrlf=false','-c','core.eol=lf','-C',str(root)]
    subprocess.run([*git,'apply','--check',str(patch)],check=True)
    subprocess.run([*git,'apply',str(patch)],check=True)
    target.write_bytes(target.read_bytes().replace(b'\r\n',b'\n'))
assert sha(target)==after
print('NINE_HOST_FP_FUSION_OPTION_PATCH_MATCHES_QUALIFIED_SOURCE',after)
