"""Negative tests ensure missing assets and wrong media cannot become a pass."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import numpy as np
import imageio.v2 as imageio
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from preflight import check,inspect_checkpoint
from validate_output import validate


def run(comparator):
    checks=[]
    with tempfile.TemporaryDirectory() as t:
        root=Path(t)
        assert check(root,device_check=False)['status']=='BLOCKED';checks.append('missing weights block execution')
        header=json.dumps({'weight':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]}}).encode()
        (root/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'\0'*4)
        r=inspect_checkpoint(root);assert r['weight_bytes']==4 and r['tensor_count']==1
        checks.append('safetensors header accounting without model allocation')
        (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'missing':'model.safetensors'}}))
        try:inspect_checkpoint(root)
        except ValueError:checks.append('checkpoint index mismatch rejected')
        else:raise AssertionError('malformed index accepted')
        def video(name,size=(64,64),fps=24,frames=8):
            p=root/name
            with imageio.get_writer(p,fps=fps,codec='libx264') as w:
                for i in range(frames):
                    a=np.zeros((*size,3),dtype=np.uint8);a[...,0]=i*20;a[:,:size[1]//2,1]=100;w.append_data(a)
            return p
        reference=video('reference.mp4')
        assert validate(reference,reference,comparator,100)['status']=='PASS';checks.append('identical synthetic video passes')
        for name,kwargs,field in [('fps',{'fps':30},'fps'),('length',{'frames':7},'decoded_frames'),('size',{'size':(32,32)},'width')]:
            r=validate(reference,video(name+'.mp4',**kwargs),comparator,0)
            assert r['status']=='FAIL' and not r['checks'][field];checks.append(name+' mismatch fails even with zero SSIM threshold')
    return dict(status='PASS',synthetic_media_only=True,full_fl2va_tested=False,checks=checks)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--comparator',type=Path,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=run(a.comparator);Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r))
