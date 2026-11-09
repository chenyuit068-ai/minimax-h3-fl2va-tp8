"""Budgeted FP32 online-attention screening; isolated final timing is separate."""
import argparse,gc,hashlib,json,statistics,sys,time
from pathlib import Path
import torch
p=argparse.ArgumentParser()
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
sys.path.insert(0,str(a.runtime))
import h3_attention_matrix as baseline
import h3_attention_online_candidate_20261023 as template
import h3_attention_online_tiled_20261023 as factory
from h3_prepared_launch import PreparedNineToothed
torch.set_num_threads(4)
report=dict(status='RUNNING',target_dcu=True,full_fl2va_tested=False,
    numerical_precision='Unchanged FP32 IEEE online application and software exp2',
    timing_scope='Preliminary screening while CPU-only independent compilation may coexist; no end-to-end claim',
    baseline_sha256=hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),
    template_sha256=hashlib.sha256(Path(template.__file__).read_bytes()).hexdigest(),
    factory_sha256=hashlib.sha256(Path(factory.__file__).read_bytes()).hexdigest(),cases=[])
started=time.monotonic()
def save():a.output.write_text(json.dumps(report,indent=2))
save()
try:
    configs=((32,32,8),(32,16,8),(16,32,8),(16,16,4),(32,32,2))
    for n in (39779,):
        gen=torch.Generator().manual_seed(375+n)
        cpu=[torch.randn((1,2,n,128),generator=gen).bfloat16() for _ in range(3)]
        queries=sorted({0,31,32,n//2,n-1,*[x for k in range(4096,n,4096) for x in (k-1,k)]})
        expected=(torch.softmax(cpu[0][:,:,queries].double()@cpu[1].double().transpose(-1,-2)*128**-.5,-1)@cpu[2].double()).float()
        tensors=[x.cuda() for x in cpu]
        original=torch.empty_like(tensors[0]);left=baseline.ChunkedMatrixAttention(*tensors,original)
        for config in configs:
            if time.monotonic()-started>300:raise TimeoutError('Five-minute online screening budget')
            row=dict(tokens=n,heads=2,head_dim=128,dtype='torch.bfloat16',tile=config,status='RUNNING')
            before=time.monotonic();right=None;actual=None
            try:
                handle=factory.make(*config)
                actual=torch.empty_like(original);right=PreparedNineToothed(handle,*tensors,actual)
                for repeat in range(2):
                    for x,y in zip(tensors,cpu):x.copy_(y if repeat==0 else -y)
                    left(*tensors,original);right(*tensors,actual);torch.cuda.synchronize()
                    assert torch.isfinite(actual).all().item()
                    torch.testing.assert_close(actual.float(),original.float(),atol=.004,rtol=.004)
                    oracle=expected if repeat==0 else -expected
                    selected=actual[:,:,queries].cpu().float()
                    torch.testing.assert_close(selected,oracle,atol=.004,rtol=.004)
                groups=[]
                for _ in range(3):
                    begin=time.perf_counter();right(*tensors,actual);torch.cuda.synchronize()
                    groups.append((time.perf_counter()-begin)*1000)
                row.update(status='PASS',groups_ms=groups,median_ms=statistics.median(groups),
                    all_output_max_abs=(actual.float()-original.float()).abs().max().item(),
                    sampled_cpu_fp64_max_abs=(selected-oracle).abs().max().item(),
                    sampled_queries=queries,changed_current_inputs=2)
            except Exception as error:
                row.update(status='FAIL',error=type(error).__name__+': '+str(error))
            row['wall_seconds_including_compilation']=time.monotonic()-before
            report['cases'].append(row);save();print(json.dumps(row),flush=True)
            del right,actual;gc.collect();torch.cuda.empty_cache()
        del left,original,tensors,cpu,expected
        gc.collect();torch.cuda.empty_cache()
    report['status']='SCREENING_COMPLETED'
except BaseException as error:
    report.update(status='FAIL_OR_BUDGET_REACHED',error=type(error).__name__+': '+str(error));raise
finally:
    report['wall_seconds']=time.monotonic()-started;save()
