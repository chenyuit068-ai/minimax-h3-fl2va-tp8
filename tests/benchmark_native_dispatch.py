"""Measure the real C++ bridge including dispatch overhead, no model-speed claim."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
import infinicore as core
from infinilm.lib import _infinilm as native
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from h3_nine_runtime import NativeFusions


def timing(fn):
    for _ in range(10):fn()
    core.sync_device()
    values=[]
    for _ in range(5):
        start=time.perf_counter()
        for _ in range(100):fn()
        core.sync_device()
        values.append((time.perf_counter()-start)*1000/100)
    return dict(median_ms=sorted(values)[2],groups_ms=values,includes_host_and_allocations=True)


def run():
    torch.manual_seed(375);cases=[]
    for dtype in (torch.float16,torch.bfloat16):
        for n in (128,512,1024,4096):
            for op in ('adaln','residual'):
                x=torch.randn((1,n,5376),device='cuda',dtype=dtype)
                mod=torch.randn((1,n,6*5376),device='cuda',dtype=dtype)
                a=mod[...,5376:2*5376];b=mod[...,3*5376:4*5376] if op=='adaln' else torch.randn_like(x)
                cdtype=core.float16 if dtype==torch.float16 else core.bfloat16
                def wrap(t):return core.strided_from_blob(t.data_ptr(),list(t.shape),list(t.stride()),dtype=cdtype,device=core.device('cuda',0))._underlying
                args=tuple(wrap(t) for t in (x,a,b));torch.cuda.synchronize()
                def execute():return native._h3_test_dispatch(op,*args)
                baseline=execute();core.sync_device()
                expected=torch.from_dlpack(native._h3_to_dlpack(baseline)).clone();torch.cuda.synchronize()
                original=timing(execute)
                with NativeFusions() as runtime:
                    fused=execute();core.sync_device()
                    torch.testing.assert_close(torch.from_dlpack(native._h3_to_dlpack(fused)),expected,rtol=0,atol=0)
                    optimized=timing(execute)
                    row=dict(dtype=str(dtype),tokens=n,hidden=5376,op=op,baseline=original,fused=optimized,
                             speedup=original['median_ms']/optimized['median_ms'],exact=True,
                             fusion_applied=runtime.calls[op]>0)
                    cases.append(row);print(json.dumps(row),flush=True)
                del x,mod,a,b,args,baseline,expected,fused
                core.sync_device();torch.cuda.empty_cache()
    return dict(status='PASS',device=torch.cuda.get_device_name(0),hip=torch.version.hip,
                synthetic=True,full_fl2va_tested=False,cases=cases)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    r=run();Path(a.output).write_text(json.dumps(r,indent=2));print('NATIVE_BENCHMARK_PASS',flush=True)
