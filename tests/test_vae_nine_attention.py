"""Independent CPU FP64 checks of the actual VAE module dispatch hook."""
import argparse,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from h3_vae_attention import VideoVAEAttention
from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as module
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
result=dict(status='RUNNING',full_fl2va_tested=False,cases=[]);started=time.monotonic()
def save():a.output.write_text(json.dumps(result,indent=2))
original=module.dispatch_attention_fn
try:
    def forbidden(*args,**kwargs):raise AssertionError('External attention fallback called')
    module.dispatch_attention_fn=forbidden
    with VideoVAEAttention() as nine:
        for dtype in (torch.float16,torch.bfloat16):
            for batch,n,h in ((1,32,2),(2,129,2),(1,1284,32),(1,1796,32)):
                for repeat in range(2):
                    generator=torch.Generator().manual_seed(375+n+repeat*97)
                    cpu=[torch.randn((batch,n,h,64),generator=generator).to(dtype) for _ in range(3)]
                    tensors=[v.to('cuda') for v in cpu]
                    with torch.inference_mode():actual=module.dispatch_attention_fn(*tensors,attn_mask=None).cpu().float()
                    heads=[0,h-1];queries=sorted(set((0,n//2,n-1)))
                    q,k,v=[t.permute(0,2,1,3)[:,heads].double() for t in cpu]
                    expected=torch.softmax(q[:,:,queries]@k.transpose(-1,-2)/8,-1)@v
                    selected=actual.permute(0,2,1,3)[:,heads][:,:,queries]
                    tolerance=.004 if dtype==torch.bfloat16 else .0006
                    assert torch.isfinite(actual).all().item()
                    torch.testing.assert_close(selected,expected.float(),atol=tolerance,rtol=tolerance)
                    row=dict(batch=batch,tokens=n,heads=h,head_dim=64,dtype=str(dtype),repeat=repeat,
                        independent_cpu_fp64=True,max_abs=(selected-expected.float()).abs().max().item(),
                        all_output_finite=True,status='PASS')
                    result['cases'].append(row);save();print(json.dumps(row),flush=True)
        result['peak_workspace_bytes']=nine.peak_workspace_bytes
    assert module.dispatch_attention_fn is forbidden,'Original module dispatch was not restored'
    result.update(status='PASS',wall_seconds=time.monotonic()-started,external_attention_calls=0,
                  module_dispatch_restored=True)
except BaseException as error:
    result.update(status='FAIL',error=f'{type(error).__name__}: {error}');raise
finally:
    module.dispatch_attention_fn=original;save()
