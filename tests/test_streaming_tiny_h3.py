"""Synthetic full-block native H3: original resident weights vs mapped streaming."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import torch
from safetensors.torch import save_file
import infinicore as core
from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner,MiniMaxH3TransformerInputs
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from h3_mapped_loader import load_mapped_state_dict
from h3_nine_runtime import NativeFusions


def run(matrix_attention=False):
    cfg=dict(model_type='minimax_h3',torch_dtype='bfloat16',hidden_size=128,num_layers=1,
             token_refiner_num_layers=1,num_attention_heads=1,attention_head_dim=128,
             ffn_hidden_size=256,latents_dim=24,audio_latents_dim=32,patch_size=[1,2,2],
             text_dim=64,timestep_input_dim=16,time_embed_hidden_size=128,time_embed_dim=64,
             rope_inv_freq_len=4,rope_freq_dim=4,norm_eps=1e-5,qk_norm_eps=1e-5,final_norm_eps=1e-5)
    generator=torch.Generator(device='cpu').manual_seed(375)
    old=os.environ.pop('INFINICORE_DEFER_PARAMETER_STORAGE',None)
    try:
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);(path/'config.json').write_text(json.dumps(cfg))
            resident=MiniMaxH3TransformerRunner(path,load_weights=False)
            weights={}
            for name,param in resident.engine.state_dict()[0].items():
                dtype=core.utils.to_torch_dtype(core.dtype(param.dtype))
                value=torch.randn(tuple(param.shape),generator=generator,dtype=torch.float32).mul_(.01).to(dtype)
                if 'norm' in name and value.ndim==1:value.fill_(1)
                weights[name]=value
            save_file(weights,str(path/'model.safetensors'))
            resident.engine.load_state_dict({k:core.from_torch(v) for k,v in weights.items()},strict=True)
            core.sync_device();resident.engine.process_weights_after_loading()
            nv,na,nt=(1024 if matrix_attention else 128),4,4;n=nv+na+nt
            def rand(tokens,channels):
                return torch.randn((1,tokens,channels),generator=generator).to(device='cuda',dtype=torch.bfloat16)
            indices=torch.arange(n,device='cuda')
            tags=torch.cat([torch.zeros(nv),torch.ones(na),torch.full((nt,),2)]).to(device='cuda',dtype=torch.int64)
            inputs=MiniMaxH3TransformerInputs(rand(nv,96),rand(na,32),rand(nt,64),
                torch.tensor([.5],device='cuda'),torch.zeros(n,device='cuda',dtype=torch.int64),tags,
                torch.stack([indices.float(),indices.float()*0,indices.float()*0],dim=-1),
                indices[:nv],indices[nv:nv+na],indices[nv+na:])
            with NativeFusions(attention=True,matrix_attention=matrix_attention):
                output=resident(inputs);core.sync_device()
                expected=(output.sample.clone(),output.audio_sample.clone())
            del resident,output,weights;gc.collect();torch.cuda.empty_cache()
            os.environ['INFINICORE_DEFER_PARAMETER_STORAGE']='1'
            streamed=MiniMaxH3TransformerRunner(path,load_weights=False)
            audit=load_mapped_state_dict(streamed.engine,path)
            gc.collect() # The engine must own CPU views after loader return.
            with NativeFusions(attention=True,matrix_attention=matrix_attention):
                for _ in range(2):
                    actual=streamed(inputs);core.sync_device()
                    torch.testing.assert_close(actual.sample,expected[0],atol=0,rtol=0)
                    torch.testing.assert_close(actual.audio_sample,expected[1],atol=0,rtol=0)
            del streamed,actual;gc.collect()
        return dict(status='PASS',synthetic_reduced_architecture=True,full_fl2va_tested=False,
                    official_weights_loaded=False,exact_resident_vs_streaming=True,
                    forward_replays=2,weight_load=audit,device=torch.cuda.get_device_name(0),matrix_attention=matrix_attention,
                    video_tokens=nv)
    finally:
        if old is None:os.environ.pop('INFINICORE_DEFER_PARAMETER_STORAGE',None)
        else:os.environ['INFINICORE_DEFER_PARAMETER_STORAGE']=old

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--matrix',action='store_true');a=p.parse_args()
    result=run(a.matrix);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
