"""Synthetic Qwen conditioner: resident and deferred weights, text and image."""
import argparse,faulthandler,gc,json,os,sys,tempfile
from pathlib import Path
import torch
from safetensors.torch import save_file
import infinicore as core
from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from h3_mapped_loader import load_mapped_state_dict
from h3_phase_pipeline import deferred_parameters
from h3_nine_runtime import NativeFusions

class TinyQwen(MiniMaxH3ConditionEncoder):
    num_hidden_layers=1

def run(output,matrix_attention=False):
    faulthandler.enable(all_threads=True)
    result=dict(status='RUNNING',synthetic_reduced_architecture=True,official_weights_loaded=False,full_fl2va_tested=False,matrix_attention=matrix_attention)
    def stage(value):
        result['stage']=value;output.write_text(json.dumps(result,indent=2));print('TINY_QWEN_STAGE',value,flush=True)
    config=dict(model_type='qwen3_vl',architectures=['Qwen3VLForConditionalGeneration'],image_token_id=501,video_token_id=502,
        vision_start_token_id=498,vision_end_token_id=499,tie_word_embeddings=False,
        text_config=dict(model_type='qwen3_vl_text',dtype='bfloat16',vocab_size=512,hidden_size=128,intermediate_size=256,
            num_hidden_layers=2,num_attention_heads=1,num_key_value_heads=1,head_dim=128,max_position_embeddings=128,
            hidden_act='silu',rms_norm_eps=1e-6,rope_theta=5000000,attention_bias=False,bos_token_id=1,eos_token_id=2,
            rope_scaling=dict(mrope_interleaved=True,mrope_section=[24,20,20],rope_type='default')),
        vision_config=dict(model_type='qwen3_vl',depth=1,hidden_size=128,hidden_act='gelu_pytorch_tanh',intermediate_size=256,
            num_heads=1,in_channels=3,num_position_embeddings=16,out_hidden_size=128,patch_size=2,spatial_merge_size=2,
            temporal_patch_size=2,deepstack_visual_indexes=[]))
    old=os.environ.pop('INFINICORE_DEFER_PARAMETER_STORAGE',None)
    generator=torch.Generator().manual_seed(375)
    try:
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'config.json').write_text(json.dumps(config))
            stage('construct_resident');resident=TinyQwen(root,load_weights=False,max_cache_len=32)
            weights={}
            for name,param in resident.engine.state_dict()[0].items():
                dtype=core.utils.to_torch_dtype(core.dtype(param.dtype))
                value=torch.randn(tuple(param.shape),generator=generator).mul_(.01).to(dtype)
                if 'norm' in name and value.ndim==1:value.fill_(1)
                weights[name]=value
            save_file(weights,str(root/'model.safetensors'))
            resident.engine.load_state_dict({k:core.from_torch(v) for k,v in weights.items()},strict=True)
            core.sync_device();resident.engine.process_weights_after_loading()
            text=dict(input_ids=torch.tensor([[1,3,4,5,6,7,8,2]]),mm_token_type_ids=torch.zeros((1,8),dtype=torch.int64),attention_mask=torch.ones((1,8),dtype=torch.int64))
            image=dict(input_ids=torch.tensor([[1,501,501,501,501,2]]),mm_token_type_ids=torch.tensor([[0,1,1,1,1,0]]),
                attention_mask=torch.ones((1,6),dtype=torch.int64),image_grid_thw=torch.tensor([[1,4,4]]),
                pixel_values=torch.randn((16,24),generator=generator,dtype=torch.bfloat16))
            expected=[]
            for name,inputs in [('text',text),('image',image)]:
                stage('resident_'+name)
                with NativeFusions(attention=True,matrix_attention=matrix_attention):value=resident.encode(inputs);core.sync_device()
                assert torch.isfinite(value).all();expected.append(value.cpu())
            del resident,weights,value;gc.collect();torch.cuda.empty_cache()
            stage('construct_streaming')
            with deferred_parameters():
                streamed=TinyQwen(root,load_weights=False,max_cache_len=32)
                audit=load_mapped_state_dict(streamed.engine,root)
            result['weight_load']=audit
            for repeat in range(2):
                for i,(name,inputs) in enumerate([('text',text),('image',image)]):
                    stage(f'streaming_{name}_{repeat}')
                    with NativeFusions(attention=True,matrix_attention=matrix_attention):actual=streamed.encode(inputs);core.sync_device()
                    torch.testing.assert_close(actual.cpu(),expected[i],atol=0,rtol=0)
            result.update(status='PASS',exact_resident_vs_streaming=True,forward_replays=2,device=torch.cuda.get_device_name(0))
            stage('finished')
    finally:
        if old is None:os.environ.pop('INFINICORE_DEFER_PARAMETER_STORAGE',None)
        else:os.environ['INFINICORE_DEFER_PARAMETER_STORAGE']=old
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--matrix',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.output,a.matrix)),flush=True)
