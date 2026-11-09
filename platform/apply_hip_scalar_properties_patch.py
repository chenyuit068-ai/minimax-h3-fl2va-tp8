"""Apply the qualified typed HIP host property adapter; no device kernels."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--site-packages',type=Path,required=True)
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    manifest=json.loads((here/'compiler-reproduction-manifest.json').read_text())
    root=args.site_packages.resolve()
    expected={'driver.py':manifest['qualified_compiler_sources']['triton_hip_driver'],
              'driver.c':manifest['qualified_compiler_sources']['triton_hip_abi']}
    paths={name:root/'triton/backends/hcu'/name for name in expected}
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    actual={name:sha(path) for name,path in paths.items()}
    if actual==expected:
        print('HIP_SCALAR_HOST_PROPERTIES_ALREADY_MATCH_QUALIFIED_SOURCE')
        return
    if actual!=manifest['hip_scalar_properties_before']:
        raise RuntimeError('Unexpected or partly patched vendor host sources; files unchanged')
    patch=here/'hip-scalar-device-properties.patch'
    if sha(patch)!=manifest['patches'][patch.name]:
        raise RuntimeError('Host patch checksum differs; files unchanged')
    for path in paths.values():
        backup=path.with_name(path.name+'.h3-pre-scalar')
        if backup.exists() and sha(backup)!=sha(path):raise RuntimeError('Existing host backup differs')
        if not backup.exists():backup.write_bytes(path.read_bytes())
    git=['git','-c','core.autocrlf=false','-c','core.eol=lf','-C',str(root)]
    env=dict(os.environ,GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull)
    subprocess.run([*git,'apply','--check',str(patch)],env=env,check=True)
    subprocess.run([*git,'apply',str(patch)],env=env,check=True)
    for name,path in paths.items():
        path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
        if sha(path)!=expected[name]:raise RuntimeError('Patched host source identity differs')
    compile(paths['driver.py'].read_text(),str(paths['driver.py']),'exec')
    print('HIP_SCALAR_HOST_PROPERTIES_MATCH_QUALIFIED_SOURCE',json.dumps(expected,sort_keys=True))

if __name__=='__main__':main()
