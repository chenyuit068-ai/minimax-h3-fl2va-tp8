"""Budgeted target-only tile/stage screening; does not alter inference sources."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import torch
import ninetoothed as nt
from ninetoothed import Tensor

parser = argparse.ArgumentParser()
parser.add_argument('--runtime', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
a = parser.parse_args()
assert not a.output.exists()
sys.path.insert(0, str(a.runtime))
import h3_attention_matrix as base
assert torch.version.hip and torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx906')
torch.set_num_threads(4)
started = time.monotonic()
report = dict(status='RUNNING', target_dcu=True, production_runtime_changed=False,
              full_fl2va_tested=False, model_computation_executed=False,
              source_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
              device=torch.cuda.get_device_name(0), hip=torch.version.hip, cases=[])

def save():
    a.output.write_text(json.dumps(report, indent=2))

def kernel(m, n, k, warps):
    def arrange(x, y, z):
        z = z.tile((m, n))
        x = x.tile((m, k)).tile((1, -1)).expand((-1, z.shape[1])); x.dtype = x.dtype.squeeze(0)
        y = y.tile((k, n)).tile((-1, 1)).expand((z.shape[0], -1)); y.dtype = y.dtype.squeeze(1)
        return x, y, z
    return nt.make(arrange, base.mm_application, tuple(Tensor(2) for _ in range(3)),
                   num_warps=warps, num_stages=1, max_num_configs=1)

save()
try:
    for label, shape in [('qk', (4096, 128, 40960)), ('pv', (4096, 40960, 128))]:
        m, k, n = shape
        gen = torch.Generator().manual_seed(375+k+n)
        cpu_x = torch.randn((m, k), generator=gen)/k**0.5
        cpu_y = torch.randn((k, n), generator=gen)
        x = cpu_x.cuda(); y = cpu_y.cuda(); z = torch.empty((m, n), device='cuda')
        rows = [0, 37, m-1]; cols = [0, 73, n-1]
        expected = cpu_x[rows].double() @ cpu_y[:, cols].double()
        for config in [(32,64,32,4),(64,64,32,4),(32,128,32,4),
                       (64,128,32,8),(64,64,64,4),(128,64,32,8),(64,128,64,8)]:
            if time.monotonic()-started > 1200:
                raise TimeoutError('Twenty-minute screening budget reached')
            row = dict(stage=label, shape=shape, tile=config, status='RUNNING')
            begin = time.monotonic()
            try:
                fn = kernel(*config)
                # Two launches with changed current input, independent FP64
                # sampled oracle and a full finite-value scan.
                errors = []
                for sign in (1, -1):
                    x.copy_(cpu_x*sign)
                    fn(x, y, z); torch.cuda.synchronize()
                    assert torch.isfinite(z).all().item()
                    sampled = z[rows][:, cols].cpu().double()
                    torch.testing.assert_close(sampled, expected*sign, atol=3e-5, rtol=3e-5)
                    errors.append((sampled-expected*sign).abs().max().item())
                for _ in range(2): fn(x, y, z)
                torch.cuda.synchronize()
                groups = []
                for _ in range(3):
                    t = time.perf_counter(); fn(x, y, z); torch.cuda.synchronize()
                    groups.append((time.perf_counter()-t)*1000)
                row.update(status='PASS', groups_ms=groups, median_ms=statistics.median(groups),
                           sampled_cpu_fp64_max_abs=max(errors), sampled_rows=rows, sampled_columns=cols)
            except Exception as error:
                row.update(status='FAIL', error=type(error).__name__+': '+str(error))
            row['wall_seconds_with_compilation'] = time.monotonic()-begin
            report['cases'].append(row); save(); print(json.dumps(row), flush=True)
        del cpu_x, cpu_y, x, y, z
        torch.cuda.empty_cache()
    report['status'] = 'SCREENING_COMPLETED'
except BaseException as error:
    report.update(status='FAILED_OR_BUDGET_REACHED', error=type(error).__name__+': '+str(error))
    raise
finally:
    report['wall_seconds'] = time.monotonic()-started; save()
