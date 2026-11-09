"""Original public audio VAE weights, synthetic latents; no full FL2VA claim."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time
import torch
from diffusers import AutoencoderKLMiniMaxH3Audio

EXPECTED='52c59e67ba8de5477c81bfbced0327aabf500f1bfdeefd5ee754529241cb26cb'

def run(directory):
    weights=directory/'diffusion_pytorch_model.safetensors'
    with weights.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=EXPECTED:raise ValueError('Expected the fixed original public audio VAE shard')
    model=AutoencoderKLMiniMaxH3Audio.from_pretrained(str(directory),local_files_only=True,torch_dtype=torch.bfloat16)
    model.requires_grad_(False);model.eval();model.to('cuda')
    generator=torch.Generator(device='cpu').manual_seed(375)
    latent=torch.randn((1,32,16),generator=generator).mul_(.02).to(device='cuda',dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats();outputs=[];seconds=[]
    with torch.inference_mode():
        for value in (latent,-latent):
            torch.cuda.synchronize();started=time.perf_counter()
            waveform=model.decode(value,return_dict=False)[0]
            torch.cuda.synchronize();seconds.append(time.perf_counter()-started)
            output=waveform.detach().float().cpu()
            assert output.ndim==3 and output.shape[:2]==(1,1) and output.shape[-1]>0
            assert torch.isfinite(output).all()
            outputs.append(output)
    assert not torch.equal(outputs[0],outputs[1]),'Fresh inputs did not change output'
    source=Path(inspect.getfile(AutoencoderKLMiniMaxH3Audio))
    return dict(status='PASS',official_component_weights_loaded=True,synthetic_latents=True,
                full_fl2va_tested=False,weight_sha256=digest,dtype='bfloat16',
                waveform_shape=list(outputs[0].shape),sampling_rate=model.config.sampling_rate,
                seconds_including_first_call=seconds,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                device=torch.cuda.get_device_name(0),reference_component_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-path',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.model_path);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
