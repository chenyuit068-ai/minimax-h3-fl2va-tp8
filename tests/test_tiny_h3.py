"""Synthetic small H3 architecture integration test, NOT the competition model."""
import argparse
import faulthandler
import json
from pathlib import Path
import sys
import tempfile
import torch
import infinicore as core
from infinilm.lib import _infinilm as native
from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner, MiniMaxH3TransformerInputs
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from h3_nine_runtime import NativeFusions


def run():
    faulthandler.enable()
    torch.manual_seed(375)
    cfg=dict(model_type='minimax_h3',torch_dtype='bfloat16',hidden_size=128,num_layers=1,
             token_refiner_num_layers=1,num_attention_heads=1,attention_head_dim=128,
             ffn_hidden_size=256,latents_dim=24,audio_latents_dim=32,patch_size=[1,2,2],
             text_dim=64,timestep_input_dim=16,time_embed_hidden_size=128,time_embed_dim=64,
             rope_inv_freq_len=4,rope_freq_dim=4,norm_eps=1e-5,qk_norm_eps=1e-5,final_norm_eps=1e-5)
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp);(p/'config.json').write_text(json.dumps(cfg))
        runner=MiniMaxH3TransformerRunner(p,device='nvidia',load_weights=False)
        print('TINY_ENGINE_INITIALIZED',flush=True)
        for name,t in runner.engine.state_dict()[0].items():
            tensor=torch.from_dlpack(native._h3_to_dlpack(t))
            if 'norm' in name and tensor.ndim==1:tensor.fill_(1)
            else:tensor.normal_(mean=0,std=0.01)
        torch.cuda.synchronize();runner.engine.process_weights_after_loading()
        print('TINY_WEIGHTS_READY',flush=True)
        nv,na,nt=128,4,4;n=nv+na+nt
        def rand(tokens,channels):return torch.randn((1,tokens,channels),device='cuda',dtype=torch.bfloat16)
        tags=torch.cat([torch.zeros(nv),torch.ones(na),torch.full((nt,),2)]).to(device='cuda',dtype=torch.int64)
        indices=torch.arange(n,device='cuda')
        inputs=MiniMaxH3TransformerInputs(rand(nv,96),rand(na,32),rand(nt,64),
            torch.tensor([0.5],device='cuda'),torch.zeros(n,device='cuda',dtype=torch.int64),tags,
            torch.stack([indices.float(),indices.float()*0,indices.float()*0],dim=-1),
            indices[:nv],indices[nv:nv+na],indices[nv+na:])
        # Both runs use the same Nine attention compatibility backend. Only
        # AdaLN/residual differ, so the reference isolates pointwise fusion.
        with NativeFusions(pointwise=False,attention=True):
            baseline=runner(inputs);core.sync_device();expected=(baseline.sample.clone(),baseline.audio_sample.clone())
        print('TINY_POINTWISE_REFERENCE_READY',flush=True)
        torch.cuda.synchronize()
        with NativeFusions(attention=True) as runtime:
            actual=runner(inputs);core.sync_device()
            torch.testing.assert_close(actual.sample,expected[0],rtol=0,atol=0)
            torch.testing.assert_close(actual.audio_sample,expected[1],rtol=0,atol=0)
            assert runtime.calls['attention']>=2,dict(runtime.calls)
            if not torch.version.hip:
                assert runtime.calls['adaln']>=2 and runtime.calls['residual']>=2,dict(runtime.calls)
            # The HIP policy deliberately keeps small pointwise tensors on Core.
            # Large pointwise fusions are exercised separately by native dispatch tests.
            calls=dict(runtime.calls)
    return dict(status='PASS',synthetic_reduced_architecture=True,official_weights_loaded=False,
                full_fl2va_tested=False,device=torch.cuda.get_device_name(0),fusion_calls=calls,
                video_latents_shape=list(actual.sample.shape),audio_latents_shape=list(actual.audio_sample.shape),
                exact_vs_native_pointwise_reference=True,
                pointwise_fusions_exercised=calls.get('adaln',0)>0 and calls.get('residual',0)>0,
                shared_attention_backend='NineToothed FP32 staged two-pass softmax' if torch.version.hip else 'NineToothed IEEE online softmax')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    r=run();Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
