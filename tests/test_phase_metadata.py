"""Construct the reference modular pipeline with lazy models; no forward pass."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from h3_phase_pipeline import make_pipeline

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--infinilm-root',type=Path,required=True)
    p.add_argument('--model-path',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('h3_phase_reference',a.infinilm_root/'examples/minimax_h3.py')
    entry=importlib.util.module_from_spec(spec);sys.modules[spec.name]=entry;spec.loader.exec_module(entry)
    paths=entry.ModelPaths(modular_dir=a.model_path,partition_dir=a.model_path)
    pipe,text,transformer=make_pipeline(entry,paths,[0],entry.TimingReport([0]),metadata_only=True)
    assert pipe._h3_phase_residency.active_model is None
    assert not pipe._h3_phase_residency.transitions
    assert str(pipe._execution_device)=='cuda:0'
    assert pipe.vae_spatial_compression_ratio==16 and pipe.vae_latents_per_chunk==5
    assert pipe.audio_latent_channels==32 and pipe.patch_size==(1,2,2)
    result=dict(status='PASS',metadata_only=True,model_forward_tested=False,full_fl2va_tested=False,
                execution_device=str(pipe._execution_device),components=list(pipe.components),
                lazy_native_and_vae_components=True,vae_spatial_compression_ratio=pipe.vae_spatial_compression_ratio)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
