"""Reconstruct the measured R4 source tree without requesting devices."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
here=Path(__file__).resolve().parent
source=here.parents[2]
if a.output.resolve().is_relative_to(source):raise ValueError('Output must be outside the source package')
if a.output.exists():raise FileExistsError('Use a new directory')
shutil.copytree(source,a.output,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
for name in ('h3_attention_native_qk.py','h3_nine_runtime.py'):
 shutil.copyfile(here.parent/'r3-runtime'/name,a.output/'runtime'/name)
shutil.copyfile(here/'h3_attention_matrix.py',a.output/'runtime/h3_attention_matrix.py')
shutil.copyfile(here/'attention-backend-manifest.json',a.output/'attention-backend-manifest.json')
sys.path.insert(0,str(a.output/'runtime'))
from h3_source_audit import inference_sources
assert inference_sources(a.output)==json.loads((here/'inference-source-identity.json').read_text())
manifest=a.output/'content-manifest.json'
if manifest.exists():
 record=json.loads(manifest.read_text())
 record.update(candidate='R4-experimental',inference_sources=inference_sources(a.output),status='EXPERIMENTAL_SOURCE_ONLY_NOT_FINAL_SUBMISSION')
 record['files']={p.relative_to(a.output).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(a.output.rglob('*')) if p.is_file() and p!=manifest and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
 manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print('R4 experimental inference sources match all 39 measured hashes; no GPU or model executed')
