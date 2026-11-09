"""Independent FP64 checks of FP32 factor application inside a Nine PV dot."""
import argparse,gc,hashlib,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
sys.path.insert(0,str(a.runtime))
import h3_pv_fused_ninetoothed_20261023 as candidate
from h3_prepared_launch import PreparedNineToothed
torch.set_num_threads(4)
report=dict(status='RUNNING',target_dcu=True,full_fl2va_tested=False,
    source_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),cases=[])
try:
    for rows,keys in ((33,127),(37,4097),(7,8193)):
        for strided in (False,True):
            blocks=(keys+4095)//4096
            for repeat in range(2):
                g=torch.Generator().manual_seed(375+keys+17*repeat)
                cpu_x=torch.randn((rows,keys),generator=g)/keys**.5
                cpu_y=torch.randn((keys,128),generator=g)
                cpu_f=torch.rand((rows,blocks),generator=g)*.1
                expected=((cpu_x.double()*cpu_f.double()[:,torch.arange(keys)//4096])@cpu_y.double()).float()
                def upload(value):
                    if not strided:return value.cuda()
                    storage=torch.empty((value.shape[0],value.shape[1]*2),device='cuda')
                    view=storage[:,::2];view.copy_(value);return view
                x,y,f=map(upload,(cpu_x,cpu_y,cpu_f));out=upload(torch.zeros((rows,128)))
                fn=PreparedNineToothed(candidate.pv_kernel,x,y,f,out)
                fn(x,y,f,out);torch.cuda.synchronize()
                actual=out.cpu()
                assert torch.isfinite(actual).all().item()
                torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
                row=dict(status='PASS',rows=rows,keys=keys,blocks=blocks,strided=strided,repeat=repeat,
                    full_cpu_fp64_max_abs=(actual-expected).abs().max().item())
                report['cases'].append(row);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
                del x,y,f,out,fn;gc.collect();torch.cuda.empty_cache()
    report['status']='PASS'
except BaseException as error:
    report.update(status='FAIL',error=type(error).__name__+': '+str(error));raise
finally:a.output.write_text(json.dumps(report,indent=2))
