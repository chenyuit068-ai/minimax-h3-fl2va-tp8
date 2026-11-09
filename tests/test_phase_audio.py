"""Actual public audio weights through lazy load/release/reload, synthetic input."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from diffusers import AutoencoderKLMiniMaxH3Audio
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from h3_phase_pipeline import PhaseResidency,LazyVAE

def run(directory):
    with (directory/'diffusion_pytorch_model.safetensors').open('rb') as source:
        assert hashlib.file_digest(source,'sha256').hexdigest()=='52c59e67ba8de5477c81bfbced0327aabf500f1bfdeefd5ee754529241cb26cb'
    phases=PhaseResidency()
    def load():
        model=AutoencoderKLMiniMaxH3Audio.from_pretrained(str(directory),local_files_only=True,torch_dtype=torch.bfloat16)
        # Match examples/minimax_h3.py, which explicitly casts both VAEs
        # after load_components. from_pretrained alone preserves some FP32 modules.
        model.requires_grad_(False);model.eval();return model.to(device='cuda',dtype=torch.bfloat16)
    phases.register('audio_vae',load)
    proxy=LazyVAE('audio_vae',directory,phases,'cuda',torch.bfloat16)
    generator=torch.Generator(device='cpu').manual_seed(375)
    latent=torch.randn((1,32,16),generator=generator).mul_(.02).to(device='cuda',dtype=torch.bfloat16)
    with torch.inference_mode():
        expected=proxy.decode(latent,return_dict=False)[0].cpu()
        torch.cuda.synchronize();loaded_bytes=torch.cuda.memory_allocated()
        phases.release();released_bytes=torch.cuda.memory_allocated()
        assert loaded_bytes-released_bytes>200*1024**2,'Model allocations remained after release'
        actual=proxy.decode(latent,return_dict=False)[0].cpu()
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        phases.release()
    return dict(status='PASS',official_audio_vae_weights=True,synthetic_latents=True,
                exact_after_release_and_reload=True,full_fl2va_tested=False,
                reference_explicit_bfloat16_cast=True,
                device=torch.cuda.get_device_name(0),loaded_torch_bytes=loaded_bytes,
                released_torch_bytes=released_bytes,phase_load_count=len(phases.transitions))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-path',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.model_path);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
