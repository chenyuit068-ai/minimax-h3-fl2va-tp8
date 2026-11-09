"""Allocated-eight-DCU tests only; reduced models are never competition outputs."""
import argparse
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))


def synchronize_all():
    import torch
    for index in range(8):torch.cuda.synchronize(index)


def collectives():
    import torch
    import torch.distributed as dist
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=8:raise RuntimeError('Collective test requires eight processes')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=90))
    try:
        for iteration in range(3):
            value=torch.full((4096,),rank+1+iteration,device=f'cuda:{rank}',dtype=torch.float32)
            dist.all_reduce(value)
            expected=36+8*iteration
            torch.testing.assert_close(value,torch.full_like(value,expected),rtol=0,atol=0)
            source=torch.full((32,),rank,device=f'cuda:{rank}',dtype=torch.int64)
            gathered=torch.empty((8*32,),device=f'cuda:{rank}',dtype=torch.int64)
            dist.all_gather_into_tensor(gathered,source)
            assert gathered.cpu().tolist()==[r for r in range(8) for _ in range(32)]
        dist.barrier();torch.cuda.synchronize(rank)
        return dict(status='PASS',world_size=8,backend='nccl_on_hip',
                    all_reduce_and_all_gather_replays=3,rank=rank,
                    full_fl2va_tested=False,official_similarity_validated=False)
    finally:dist.destroy_process_group()


def native_eight_devices():
    print('NATIVE_ATTENTION_IMPORT_BEGIN',flush=True)
    import torch
    import infinicore as core
    from infinilm.lib import _infinilm as native
    from h3_nine_runtime import NativeFusions
    print('NATIVE_ATTENTION_IMPORT_PASS',flush=True)
    cases=[]
    # Two Torch producer streams exercise explicit handoff to Core's own
    # stream. They are not claims of two distinct native execution streams.
    # Fresh tensors on the second pass must produce the new oracle outputs.
    streams={i:[torch.cuda.Stream(device=i),torch.cuda.Stream(device=i)] for i in range(8)}
    with NativeFusions(pointwise=False,attention=True,matrix_attention=True,
                       device_ids=range(8),cache_policy='pinned',max_signatures=32) as runtime:
        for repeat in range(2):
            for index in range(8):
                for n,d in ((136,72),(1025,128)):
                    print('NATIVE_ATTENTION_CASE_BEGIN',repeat,index,n,d,flush=True)
                    generator=torch.Generator().manual_seed(375+repeat*17+index+n)
                    cpu=[torch.randn((1,2,n,d),generator=generator).to(torch.bfloat16) for _ in range(3)]
                    expected=(torch.softmax(cpu[0].double()@cpu[1].double().transpose(-1,-2)*d**-.5,-1)@cpu[2].double()).float()
                    with torch.cuda.device(index):
                        core.set_device(core.device('cuda',index))
                        for stream in streams[index]:
                            # The native bridge owns the Core stream. Tensor
                            # initialization is complete before handing it off.
                            with torch.cuda.stream(stream):
                                tensors=[]
                                for value in cpu:
                                    base=torch.empty((1,n,2,3*d),device=f'cuda:{index}',dtype=torch.bfloat16)
                                    view=base[...,d:2*d];view.copy_(value.permute(0,2,1,3));tensors.append(view)
                            stream.synchronize()
                            args=tuple(core.strided_from_blob(t.data_ptr(),list(t.shape),list(t.stride()),
                                dtype=core.bfloat16,device=core.device('cuda',index))._underlying for t in tensors)
                            answer=native._h3_test_dispatch('attention',*args);core.sync_device()
                            actual=torch.from_dlpack(native._h3_to_dlpack(answer)).permute(0,2,1,3).cpu().float()
                            torch.testing.assert_close(actual,expected,rtol=.004,atol=.004)
                            cases.append(dict(device=index,tokens=n,head_dim=d,repeat=repeat,
                                max_abs=(actual-expected).abs().max().item()))
                    print('NATIVE_ATTENTION_CASE_PASS',repeat,index,n,d,flush=True)
        synchronize_all()
        assert set(runtime.calls_by_device)==set(range(8))
        assert runtime.workspace_evictions==0
        calls={str(i):dict(v) for i,v in runtime.calls_by_device.items()}
        workspace={str(i):runtime.workspace_bytes_for_device(i) for i in range(8)}
    return dict(status='PASS',world_size=8,cases=cases,attention_calls_by_device=calls,
                workspace_bytes_by_device=workspace,scratch_evictions=0,producer_streams_per_device=2,
                oracle='independent CPU FP64 unmasked attention',
                full_fl2va_tested=False,official_similarity_validated=False)


def initialize_weights(model,generator):
    import torch
    import infinicore as core
    weights={}
    for name,param in model.engine.state_dict()[0].items():
        dtype=core.utils.to_torch_dtype(core.dtype(param.dtype))
        value=torch.randn(tuple(param.shape),generator=generator).mul_(.01).to(dtype)
        if 'norm' in name and value.ndim==1:value.fill_(1)
        weights[name]=value
    load_weights(model,weights)
    return weights


def load_weights(model,weights):
    import infinicore as core
    model.engine.load_state_dict({k:core.from_torch(v) for k,v in weights.items()},strict=True)
    synchronize_all();model.engine.process_weights_after_loading();synchronize_all()


def tiny_tp8(kind):
    import torch
    import infinicore as core
    from infinilm.multimodal.minimax_h3 import (MiniMaxH3TransformerRunner,
        MiniMaxH3TransformerInputs,MiniMaxH3ConditionEncoder)
    from h3_nine_runtime import NativeFusions
    generator=torch.Generator().manual_seed(375)
    if os.environ.get('INFINICORE_DEFER_PARAMETER_STORAGE'):
        raise RuntimeError('Test native resident TP, not single-device deferred storage')
    torch.cuda.set_device(0);core.set_device(core.device('cuda',0))
    class TinyQwen(MiniMaxH3ConditionEncoder):num_hidden_layers=1
    if kind=='h3':
        config=dict(model_type='minimax_h3',torch_dtype='bfloat16',hidden_size=1024,num_layers=1,
            token_refiner_num_layers=1,num_attention_heads=8,attention_head_dim=128,
            ffn_hidden_size=2048,latents_dim=24,audio_latents_dim=32,patch_size=[1,2,2],
            text_dim=64,timestep_input_dim=16,time_embed_hidden_size=128,time_embed_dim=64,
            rope_inv_freq_len=4,rope_freq_dim=4,norm_eps=1e-5,qk_norm_eps=1e-5,final_norm_eps=1e-5)
        constructor=MiniMaxH3TransformerRunner
    else:
        config=dict(model_type='qwen3_vl',architectures=['Qwen3VLForConditionalGeneration'],
            image_token_id=501,video_token_id=502,vision_start_token_id=498,vision_end_token_id=499,
            tie_word_embeddings=False,
            text_config=dict(model_type='qwen3_vl_text',dtype='bfloat16',vocab_size=512,hidden_size=1024,
                intermediate_size=2048,num_hidden_layers=2,num_attention_heads=8,num_key_value_heads=8,
                head_dim=128,max_position_embeddings=128,hidden_act='silu',rms_norm_eps=1e-6,
                rope_theta=5000000,attention_bias=False,bos_token_id=1,eos_token_id=2,
                rope_scaling=dict(mrope_interleaved=True,mrope_section=[24,20,20],rope_type='default')),
            vision_config=dict(model_type='qwen3_vl',depth=1,hidden_size=1024,hidden_act='gelu_pytorch_tanh',
                intermediate_size=2048,num_heads=8,in_channels=3,num_position_embeddings=16,
                out_hidden_size=1024,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,
                deepstack_visual_indexes=[]))
        constructor=lambda path,**kwargs:TinyQwen(path,max_cache_len=32,**kwargs)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/'config.json').write_text(json.dumps(config))
        resident=constructor(root,tp_device_ids=[0],load_weights=False)
        weights=initialize_weights(resident,generator)
        if kind=='h3':
            inputs=[]
            for repeat in range(2):
                nv,na,nt=1024,4,4;n=nv+na+nt
                def rand(tokens,channels):
                    return torch.randn((1,tokens,channels),generator=generator).to(device='cuda:0',dtype=torch.bfloat16)
                indices=torch.arange(n,device='cuda:0')
                tags=torch.cat([torch.zeros(nv),torch.ones(na),torch.full((nt,),2)]).to(device='cuda:0',dtype=torch.int64)
                inputs.append(MiniMaxH3TransformerInputs(rand(nv,96),rand(na,32),rand(nt,64),
                    torch.tensor([.5],device='cuda:0'),torch.zeros(n,device='cuda:0',dtype=torch.int64),tags,
                    torch.stack([indices.float(),indices.float()*0,indices.float()*0],dim=-1),
                    indices[:nv],indices[nv:nv+na],indices[nv+na:]))
            def forward(model,value):
                answer=model(value);synchronize_all()
                return answer.sample.cpu().float(),answer.audio_sample.cpu().float()
        else:
            inputs=[dict(input_ids=torch.tensor([[1,3,4,5,6,7,8,2]]),
                         mm_token_type_ids=torch.zeros((1,8),dtype=torch.int64),
                         attention_mask=torch.ones((1,8),dtype=torch.int64)),
                    dict(input_ids=torch.tensor([[1,501,501,501,501,2]]),
                         mm_token_type_ids=torch.tensor([[0,1,1,1,1,0]]),
                         attention_mask=torch.ones((1,6),dtype=torch.int64),
                         image_grid_thw=torch.tensor([[1,4,4]]),
                         pixel_values=torch.randn((16,24),generator=generator,dtype=torch.bfloat16))]
            def forward(model,value):
                answer=model.encode(value);synchronize_all();return (answer.cpu().float(),)
        # The TP1 oracle shares the qualified Nine attention algorithm; the
        # independent FP64 numeric check is the separate native-device test.
        with NativeFusions(attention=True,matrix_attention=True,device_ids=[0],cache_policy='pinned'):
            expected=[forward(resident,value) for value in inputs]
        del resident;gc.collect();synchronize_all()
        tp=constructor(root,tp_device_ids=list(range(8)),load_weights=False)
        load_weights(tp,weights)
        rank_shapes=[{name:list(t.shape) for name,t in rank.items()} for rank in tp.engine.state_dict()]
        if len(rank_shapes)!=8:raise AssertionError('Engine did not create eight parameter ranks')
        cases=[]
        with NativeFusions(attention=True,matrix_attention=True,device_ids=range(8),cache_policy='pinned') as runtime:
            for repeat in range(2):
                for case,value in enumerate(inputs):
                    actual=forward(tp,value)
                    errors=[]
                    for answer,reference in zip(actual,expected[case]):
                        torch.testing.assert_close(answer,reference,rtol=.04,atol=.03)
                        errors.append((answer-reference).abs().max().item())
                    cases.append(dict(case=case,repeat=repeat,max_abs=errors,rtol=.04,atol=.03))
            synchronize_all()
            if set(runtime.calls_by_device)!=set(range(8)):
                raise AssertionError('TP did not execute Nine callbacks on every device')
            calls={str(i):dict(v) for i,v in runtime.calls_by_device.items()}
        del tp;gc.collect();synchronize_all()
    return dict(status='PASS',world_size=8,model=kind,cases=cases,
                per_rank_parameter_shapes=rank_shapes,attention_calls_by_device=calls,
                synthetic_reduced_architecture=True,official_weights_loaded=False,
                tp1_vs_tp8=True,full_fl2va_tested=False,official_similarity_validated=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test',required=True,choices=['collectives','native_eight_devices','tiny_tp8_h3','tiny_tp8_qwen'])
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose fresh test evidence')
    import faulthandler
    faulthandler.enable(all_threads=True)
    os.environ['H3_PLATFORM_QUALIFICATION_MODE']='component-tests'
    if a.test=='collectives':result=collectives()
    else:
        import torch
        if torch.cuda.device_count()!=8 or not torch.version.hip:
            raise RuntimeError('Run inside an allocated eight-DCU machine')
        result=native_eight_devices() if a.test=='native_eight_devices' else tiny_tp8(a.test.rsplit('_',1)[1])
    if int(os.environ.get('RANK','0'))==0:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result),flush=True)
