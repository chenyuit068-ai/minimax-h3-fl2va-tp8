"""Native worker-stream matrix dispatch, fresh pointers and bounded scratch cache."""
import argparse,gc,json,sys
from pathlib import Path
import torch
import infinicore as core
from infinilm.lib import _infinilm as native
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from h3_nine_runtime import NativeFusions

def run(output):
    assert torch.version.hip
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':',1)[0] in ('gfx906','gfx936')
    assert not output.exists(),'Choose fresh evidence'
    result=dict(status='RUNNING',synthetic=True,full_fl2va_tested=False,cases=[])
    output.parent.mkdir(parents=True,exist_ok=True)
    def record():output.write_text(json.dumps(result,indent=2))
    record()
    try:
        core.set_device(core.device('cuda',0))
        with NativeFusions(max_signatures=2,pointwise=False,attention=True,
                           matrix_attention=True,max_workspace_bytes=224*1024**2) as runtime:
            for n,h,d in ((136,16,72),(512,56,128),(1025,2,128),(8193,2,128)):
                queries=sorted({0,n//2,n-1,*[q for start in range(4096,n,4096) for q in (start-1,start)]})
                heads=[0,h-1]
                for dtype in (torch.float16,torch.bfloat16):
                    for repeat in range(2):
                        # Allocate fresh QKV views for every replay; the host
                        # cache must not retain earlier input or output values.
                        cpu=[torch.randn((1,h,n,d),generator=torch.Generator().manual_seed(375+n+repeat*11+i)).to(dtype) for i in range(3)]
                        tensors=[]
                        for value in cpu:
                            base=torch.empty((1,n,h,3*d),device='cuda',dtype=dtype)
                            view=base[...,d:2*d];view.copy_(value.permute(0,2,1,3));tensors.append(view)
                        q=cpu[0][:,heads][:,:,queries].double();k,v=[x[:,heads].double() for x in cpu[1:]]
                        expected=(torch.softmax(q@k.transpose(-1,-2)*d**-.5,-1)@v).float()
                        cdtype=core.float16 if dtype==torch.float16 else core.bfloat16
                        args=tuple(core.strided_from_blob(t.data_ptr(),list(t.shape),list(t.stride()),dtype=cdtype,device=core.device('cuda',0))._underlying for t in tensors)
                        torch.cuda.synchronize()
                        answer=native._h3_test_dispatch('attention',*args);core.sync_device()
                        actual=torch.from_dlpack(native._h3_to_dlpack(answer)).permute(0,2,1,3).cpu().float()
                        assert torch.isfinite(actual).all().item()
                        tolerance=.004 if dtype==torch.bfloat16 else .0006
                        selected=actual[:,heads][:,:,queries]
                        torch.testing.assert_close(selected,expected,atol=tolerance,rtol=tolerance)
                        assert runtime.workspace_bytes<=runtime.max_workspace_bytes
                        row=dict(status='PASS',tokens=n,heads=h,head_dim=d,dtype=str(dtype),repeat=repeat,
                            max_abs=(selected-expected).abs().max().item(),atol=tolerance,rtol=tolerance,
                            native_input_strided=True,current_pointer_replay=True,all_output_finite=True,
                            independent_cpu_fp64_heads=heads,independent_cpu_fp64_queries=queries,
                            workspace_bytes=runtime.workspace_bytes,cache_entries=len(runtime.cache))
                        result['cases'].append(row);record();print(json.dumps(row),flush=True)
                        del args,answer,actual,selected,tensors,base,view,cpu,q,k,v,expected
                        gc.collect();torch.cuda.empty_cache()
            assert runtime.attention_routes['matrix']==8 and runtime.attention_routes['vector']==8,dict(runtime.attention_routes)
            assert runtime.workspace_evictions>0,'The bounded cache eviction path was not exercised'
            runtime.clear_cache()
            assert not runtime.cache and not runtime.cache_streams and runtime.workspace_bytes==0
            result.update(status='PASS',checks=16,attention_routes=dict(runtime.attention_routes),
                peak_workspace_bytes=runtime.peak_workspace_bytes,workspace_budget_bytes=runtime.max_workspace_bytes,
                workspace_evictions=runtime.workspace_evictions,cache_clear_checked=True,device=torch.cuda.get_device_name(0))
    except BaseException as error:
        result.update(status='FAIL',error=f'{type(error).__name__}: {error}');raise
    finally:record()
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(run(parser.parse_args().output)),flush=True)
