"""Independent CPU FP64 qualification of an isolated candidate, not a model run."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import h3_block_softmax_candidate_20261023 as candidate

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
assert torch.version.hip
torch.set_num_threads(4)
report = dict(status='RUNNING', target_dcu=True, production_runtime_changed=False,
              full_fl2va_tested=False, source_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
              cases=[])
try:
    for n in (1, 33, 513, 4096, 4097, 8193, 39779, 40003):
        for scale in (1, 64):
            for strided in (False, True):
                gen = torch.Generator().manual_seed(375+n+scale)
                cpu = torch.randn((13, n), generator=gen)*scale
                reference = torch.softmax(cpu.double()*128**-0.5, -1).float()
                if strided:
                    storage = torch.empty((13, n+37), device='cuda')
                    scores = storage[:, 11:11+n]; scores.copy_(cpu)
                else:
                    scores = cpu.cuda()
                out = torch.empty_like(scores)
                blocks = (n+candidate.BLOCK-1)//candidate.BLOCK
                maxima = torch.empty((13, blocks), device='cuda')
                sums = torch.empty_like(maxima)
                for repeat in range(2):
                    if repeat:
                        scores.copy_(-cpu)
                        reference = torch.softmax(-cpu.double()*128**-0.5, -1).float()
                    candidate.local_kernel(scores, maxima, sums, out)
                    candidate.global_kernel(out, maxima, sums, out)
                    torch.cuda.synchronize()
                    actual = out.cpu()
                    assert torch.isfinite(actual).all()
                    torch.testing.assert_close(actual, reference, atol=2e-7, rtol=2e-6)
                    row = dict(tokens=n, scale=scale, strided=strided, repeat=repeat,
                               status='PASS', max_abs=(actual-reference).abs().max().item(),
                               row_sum_max_error=(actual.sum(-1)-1).abs().max().item())
                    report['cases'].append(row)
                    a.output.write_text(json.dumps(report, indent=2)); print(json.dumps(row), flush=True)
    report['status'] = 'PASS'
except BaseException as error:
    report.update(status='FAIL', error=type(error).__name__+': '+str(error))
    raise
finally:
    a.output.write_text(json.dumps(report, indent=2))
