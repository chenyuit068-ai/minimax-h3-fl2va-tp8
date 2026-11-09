"""Full spatial/temporal VAE component workload on synthetic latents only."""
import argparse
import gc
import json
import time
import sys
from pathlib import Path
import torch
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from h3_vae_attention import VideoVAEAttention
from h3_source_audit import inference_sources
from contextlib import nullcontext

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-path',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=args.model_path.resolve();output=args.output.resolve()
if output.exists():raise FileExistsError('Choose a new component result file')
output.parent.mkdir(parents=True,exist_ok=True)
result=dict(status='RUNNING',synthetic_latents=True,full_fl2va_tested=False,
            original_reference_bfloat16_policy=True,
            inference_sources=inference_sources(Path(__file__).resolve().parents[1]),
            torch=torch.__version__,arch=torch.cuda.get_device_properties(0).gcnArchName)

def record():
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

try:
    for name,cls,shape,expected in [
        ('vae',AutoencoderKLMiniMaxH3,(1,24,37,48,84),(1,3,124,768,1344)),
        ('audio_vae',AutoencoderKLMiniMaxH3Audio,(2,32,207),(2,1,165600))]:
        result['stage']=name;record()
        model=cls.from_pretrained(str(root/name),local_files_only=True,torch_dtype=torch.bfloat16)
        model.requires_grad_(False);model.eval();model.to(device='cuda',dtype=torch.bfloat16)
        latent=torch.randn(shape,generator=torch.Generator().manual_seed(375)).mul_(.02).to('cuda',torch.bfloat16)
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        with torch.inference_mode(), (VideoVAEAttention() if name=='vae' else nullcontext()) as attention:
            value=model.decode(latent,return_dict=False)[0]
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        assert tuple(value.shape)==expected,(name,tuple(value.shape),expected)
        chunk=2 if name=='vae' else 8192
        finite=all(torch.isfinite(value.narrow(2,start,min(chunk,value.shape[2]-start))).all().item()
                   for start in range(0,value.shape[2],chunk))
        assert finite
        result[name]=dict(status='PASS',latent_shape=list(shape),decoded_shape=list(value.shape),finite=True,
            decode_seconds=seconds,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            default_tiling=getattr(model,'use_tiling',None),
            nine_attention_calls=getattr(attention,'calls',0),
            nine_workspace_bytes=getattr(attention,'peak_workspace_bytes',0))
        record();del model,latent,value;gc.collect();torch.cuda.empty_cache()
    result['status']='COMPONENT_WORKLOAD_PASS_NOT_GENERATION_OR_QUALITY'
except BaseException as error:
    result.update(status='FAILED',error=f'{type(error).__name__}: {error}');raise
finally:record()
