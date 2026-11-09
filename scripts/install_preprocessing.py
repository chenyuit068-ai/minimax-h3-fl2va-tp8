"""Install the declared CPU preprocessing wheel without replacing vendor Torch."""
import argparse,hashlib,json,subprocess,sys,importlib
from pathlib import Path

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wheelhouse',type=Path,required=True)
    parser.add_argument('--target',type=Path,required=True);args=parser.parse_args()
    declaration=json.loads((Path(__file__).resolve().parents[1]/'auxiliary-dependencies.json').read_text())['torchvision']
    wheel=args.wheelhouse/declaration['filename']
    with wheel.open('rb') as source:assert hashlib.file_digest(source,'sha256').hexdigest()==declaration['sha256']
    import torch
    original=torch.__version__
    if original.split('+')[0]!='2.9.0':raise RuntimeError('This preprocessing wheel was validated against the provided Torch 2.9.0 environment')
    if args.target.exists() and any(args.target.iterdir()):raise FileExistsError('Use an empty isolated dependency directory')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--target',str(args.target),str(wheel)],check=True)
    sys.path.insert(0,str(args.target.resolve()))
    importlib.invalidate_caches()
    import torchvision
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    assert to_tensor(Image.new('RGB',(32,32))).shape==(3,32,32)
    from transformers import Qwen3VLProcessor
    assert torch.__version__==original
    print(json.dumps(dict(status='PREPROCESSING_IMPORT_PASS',torch=original,torchvision=torchvision.__version__,target=str(args.target.resolve()),full_fl2va_tested=False)))
