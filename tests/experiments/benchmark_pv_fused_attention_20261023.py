"""Paired synchronized operator timing after all background qualification exits."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import torch

p = argparse.ArgumentParser()
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args(); assert not a.output.exists()
# This driver has not started while its shell prerequisite waits. Finish any
# budgeted screening in the same experiment directory before timing begins.
waiting_started=time.monotonic()
while any(not log.with_suffix('.exit').exists() for log in a.output.parent.glob('*screening.log')):
    if time.monotonic()-waiting_started>1200:
        raise TimeoutError('Background screening did not finish within 20 minutes')
    time.sleep(5)
waited_for_screening=time.monotonic()-waiting_started
sys.path.insert(0, str(a.runtime))
import h3_attention_matrix as baseline
import h3_attention_split_candidate_20261023 as split
import h3_attention_pv_fused_candidate_20261023 as fused
import h3_pv_fused_ninetoothed_20261023 as fused_math
from h3_prepared_launch import PreparedNineToothed
torch.set_num_threads(4)
report = dict(status='RUNNING', target_dcu=True, full_fl2va_tested=False,
    waited_for_background_screening_seconds=waited_for_screening,
    production_runtime_changed=False, official_baseline_compared=False,
    method='Two warmups, three alternating synchronized single-call groups; wall-clock median',
    sources={m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (baseline, split, fused, fused_math)},
    cases=[])
started = time.monotonic()
try:
    for n, heads in ((8193,56),(40003,2),(39779,56)):
        gen = torch.Generator().manual_seed(375+n)
        cpu = [torch.randn((1,heads,n,128), generator=gen).bfloat16() for _ in range(3)]
        tensors = [t.cuda() for t in cpu]
        outputs = {name:torch.empty_like(tensors[0]) for name in ('baseline','split','fused')}
        prepared = dict(baseline=baseline.ChunkedMatrixAttention(*tensors, outputs['baseline']),
            split=split.ChunkedMatrixAttention(*tensors, outputs['split']),
            fused=fused.ChunkedMatrixAttention( *tensors, outputs['fused']))
        for name, handle in prepared.items():
            for _ in range(2):handle(*tensors, outputs[name])
            torch.cuda.synchronize()
        for name in ('split','fused'):
            torch.testing.assert_close(outputs[name].float(), outputs['baseline'].float(), atol=.004, rtol=.004)
        groups = {name:[] for name in prepared}
        for group in range(3):
            order = list(prepared)
            if group%2: order.reverse()
            for name in order:
                if time.monotonic()-started > 1200:raise TimeoutError('20-minute timing budget reached')
                begin = time.perf_counter(); prepared[name](*tensors, outputs[name]); torch.cuda.synchronize()
                groups[name].append((time.perf_counter()-begin)*1000)
        medians = {name:statistics.median(values) for name,values in groups.items()}
        row = dict(tokens=n, heads=heads, head_dim=128, dtype='torch.bfloat16',
            groups_ms=groups, median_ms=medians,
            explicit_workspace_bytes={name:sum(t.numel()*t.element_size() for t in (*getattr(handle,"buffers",()),*(() if getattr(handle,"scores",None) is None else (handle.scores,)))) for name,handle in prepared.items()},
            speedup_over_v4={name:medians['baseline']/medians[name] for name in ('split','fused')},
            all_output_max_abs={name:(outputs[name].float()-outputs['baseline'].float()).abs().max().item() for name in ('split','fused')},
            status='PASS')
        report['cases'].append(row);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
        del prepared, outputs, tensors, cpu
        gc.collect();torch.cuda.empty_cache()
    report['status']='PASS'
except BaseException as error:
    report.update(status='FAIL', error=type(error).__name__+': '+str(error));raise
finally:
    report['wall_seconds']=time.monotonic()-started;a.output.write_text(json.dumps(report,indent=2))
