"""Independent attention oracle, tails and multiple heads; no official assets."""
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from h3_attention_ninetoothed import h3_attention,h3_attention_vector
from h3_attention_twopass import h3_attention_twopass
from h3_attention_staged import StagedAttention


def run(native_check=False,vector=False):
    staged=bool(torch.version.hip) and not vector
    torch.manual_seed(375);cases=[]
    for dtype in (torch.float16,torch.bfloat16):
        for n,heads,d in ((1,1,32),(4,2,64),(33,2,128),(136,1,128),(33,2,72),(136,16,72)):
            q,k,v=(torch.randn((1,heads,n,d)).to(dtype).cuda() for _ in range(3))
            out=torch.empty_like(q)
            kernel=StagedAttention(h3_attention_twopass,q,k,v,out) if staged else (h3_attention_vector if vector else h3_attention)
            kernel(q,k,v,out);torch.cuda.synchronize()
            # CPU FP64 oracle avoids trusting the same vendor GPU math stack
            # being adapted. Only small synthetic test inputs move to the CPU.
            q64,k64,v64=(t.cpu().double() for t in (q,k,v))
            expected=(torch.softmax((q64@k64.transpose(-1,-2))*d**-0.5,dim=-1)@v64).to(device='cuda',dtype=torch.float32)
            difference=(out.float()-expected).abs()
            tolerance=0.004 if dtype==torch.bfloat16 else 0.0006
            if not torch.allclose(out.float(),expected,rtol=tolerance,atol=tolerance):
                print('ATTENTION_NUMERICAL_FAILURE',dtype,n,heads,d,'max_abs',difference.max().item(),flush=True)
                torch.save(dict(q=q.cpu(),k=k.cpu(),v=v.cpu(),actual=out.cpu(),expected=expected.cpu()),
                           Path(__file__).with_name('attention-failure.pt'))
            torch.testing.assert_close(out.float(),expected,rtol=tolerance,atol=tolerance)
            if native_check:
                import infinicore as core
                from infinilm.lib import _infinilm as native
                from h3_nine_runtime import NativeFusions
                tensors=tuple(t.permute(0,2,1,3).contiguous() for t in (q,k,v))
                cdtype=core.float16 if dtype==torch.float16 else core.bfloat16
                args=tuple(core.strided_from_blob(t.data_ptr(),list(t.shape),list(t.stride()),
                    dtype=cdtype,device=core.device('cuda',0))._underlying for t in tensors)
                torch.cuda.synchronize()
                with NativeFusions(pointwise=False,attention=True) as runtime:
                    result=native._h3_test_dispatch('attention',*args);core.sync_device()
                    actual=torch.from_dlpack(native._h3_to_dlpack(result)).permute(0,2,1,3)
                    torch.testing.assert_close(actual.float(),expected,rtol=tolerance,atol=tolerance)
                    assert runtime.calls['attention']==1
            assert torch.isfinite(out).all()
            row=dict(dtype=str(dtype),tokens=n,heads=heads,head_dim=d,max_abs_error=difference.max().item())
            cases.append(row);print(json.dumps(row),flush=True)
    return dict(status='PASS',device=torch.cuda.get_device_name(0),hip=torch.version.hip,cases=cases,
                synthetic=True,full_fl2va_tested=False,native_bridge_checked=native_check,vector_reduction=vector,
                fp32_head_staging=staged,
                oracle='CPU FP64 matmul + softmax + matmul')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--native',action='store_true');p.add_argument('--vector',action='store_true');a=p.parse_args()
    r=run(a.native,a.vector);Path(a.output).write_text(json.dumps(r,indent=2));print('ATTENTION_CHECK_PASS',flush=True)
