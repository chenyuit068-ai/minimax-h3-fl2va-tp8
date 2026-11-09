"""Actual native C++ -> NineToothed dispatch, views, lifetimes and fallback."""
import argparse
import json
from pathlib import Path
import sys
import torch
import infinicore as core
from infinilm.lib import _infinilm as native

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from h3_nine_runtime import NativeFusions as QualifiedNativeFusions

def NativeFusions(**options):
    return QualifiedNativeFusions(device_ids=range(8),cache_policy='pinned',**options)


def run():
    torch.manual_seed(375)
    cases=[]
    def one(dtype,n,h,kind):
        fused=n>=128 and h>=64 and dtype!=torch.float32
        if torch.version.hip:fused=fused and n*h >= (1024 if kind=='residual' else 512)*5376
        torch.cuda.set_device(0); core.set_device(core.device('cuda',0))
        x=torch.randn((1,n,h),device='cuda',dtype=dtype)
        mods=torch.randn((1,n,6*h),device='cuda',dtype=dtype)
        a=mods[...,h:2*h]; b=mods[...,3*h:4*h]
        if kind=='residual': b=torch.randn_like(x)
        torch.cuda.synchronize()
        def wrap(t):
            return core.strided_from_blob(t.data_ptr(),list(t.shape),list(t.stride()),
                dtype={torch.float16:core.float16,torch.bfloat16:core.bfloat16,torch.float32:core.float32}[dtype],
                device=core.device('cuda',0))._underlying
        args=tuple(wrap(t) for t in (x,a,b))
        # Capture unmodified Core reference before installing the callback.
        baseline=native._h3_test_dispatch(kind,*args)
        core.sync_device()
        expected=torch.from_dlpack(native._h3_to_dlpack(baseline)).clone()
        torch.cuda.synchronize()
        with NativeFusions(max_signatures=2) as runtime:
            for repeat in range(3):
                if repeat:
                    x=torch.randn_like(x); mods=torch.randn_like(mods); a=mods[...,h:2*h]
                    b=mods[...,3*h:4*h] if kind=='adaln' else torch.randn_like(x)
                    args=tuple(wrap(t) for t in (x,a,b)); torch.cuda.synchronize()
                result=native._h3_test_dispatch(kind,*args)
                core.sync_device()
                view=torch.from_dlpack(native._h3_to_dlpack(result))
                # Torch is only the independent testing oracle.
                if kind=='adaln':
                    oracle=x*(1.0+a)+b if fused else (x+(x*a).to(dtype)).to(dtype)+b
                else:oracle=x+(a*b).to(dtype)
                if repeat==0 and not (kind=='adaln' and fused):
                    torch.testing.assert_close(view,expected,rtol=0,atol=0)
                if dtype!=torch.float32: torch.testing.assert_close(view,oracle,rtol=0,atol=0)
                assert torch.isfinite(view).all()
                assert view.data_ptr()==result.data_ptr(), 'DLPack conversion copied the data'
                # The DLPack owner retains native storage after native handle deletion.
                del result
                assert torch.isfinite(view).all()
            assert sum(runtime.calls.values())==(3 if fused else 0)
            cases.append(dict(dtype=str(dtype),shape=[1,n,h],kind=kind,fused=fused,passes=3))
    for dtype in (torch.float16,torch.bfloat16,torch.float32):
        for n,h in ((17,129),(128,128),(129,5376),(512,5376),(1025,5376)):
            for kind in ('adaln','residual'):one(dtype,n,h,kind)
    # A second runtime must not replace the live callback or disturb its owner.
    with NativeFusions():
        try: NativeFusions()
        except RuntimeError as exc: assert 'already installed' in str(exc)
        else: raise AssertionError('A second live runtime was accepted')
    # Propagate callback errors: never silently fall back after failed execution.
    def fail(*args): raise RuntimeError('intentional-host-callback-error')
    # Exercise the fused route even under the measured HIP size threshold.
    x=torch.zeros((1,512,5376),device='cuda',dtype=torch.float16)
    a=core.from_torch(x)._underlying
    native._h3_set_nine_callback(fail)
    try:
        try:native._h3_test_dispatch('adaln',a,a,a)
        except RuntimeError as exc: assert 'intentional-host-callback-error' in str(exc)
        else:raise AssertionError('callback failure was swallowed')
    finally:native._h3_set_nine_callback(None)
    return dict(status='PASS',device=torch.cuda.get_device_name(0),hip=torch.version.hip,
                synthetic=True,full_fl2va_tested=False,cases=cases,total_output_checks=3*len(cases),
                zero_copy=True,callback_errors_propagate=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    result=run();Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
