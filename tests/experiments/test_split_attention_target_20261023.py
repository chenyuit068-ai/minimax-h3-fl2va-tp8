"""Bounded candidate tests: whole-output old/new checks and sampled FP64 oracle."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

p = argparse.ArgumentParser()
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args(); assert not a.output.exists()
sys.path.insert(0, str(a.runtime))
import h3_attention_matrix as baseline
import h3_attention_split_candidate_20261023 as candidate
assert torch.version.hip
torch.set_num_threads(4)
report = dict(status='RUNNING', target_dcu=True, production_runtime_changed=False,
              full_fl2va_tested=False, timing_valid=False,
              reason='Correctness qualification; may overlap isolated CPU rebuild',
              candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
              baseline_sha256=hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(), cases=[])
started = time.monotonic()
try:
    shapes = [(n, 2, dtype) for n in (1, 33, 513, 1024, 8193, 39779, 40003)
              for dtype in (torch.float16, torch.bfloat16, torch.float32)]
    shapes += [(39779, 56, torch.bfloat16)]
    for n, heads, dtype in shapes:
        gen = torch.Generator().manual_seed(375+n+heads)
        cpu = [torch.randn((1, heads, n, 128), generator=gen).to(dtype) for _ in range(3)]
        queries = sorted({0, n//2, n-1, *[q for i in range(4096, n, 4096) for q in (i-1, i)]})
        selected_heads = sorted({0, heads-1})
        q = cpu[0][:, selected_heads][:, :, queries].double()
        k = cpu[1][:, selected_heads].double(); v = cpu[2][:, selected_heads].double()
        expected = (torch.softmax(q@k.transpose(-1,-2)*128**-0.5, -1)@v).float()
        del q, k, v
        for strided in (False, True):
            tensors = []
            for x in cpu:
                if strided:
                    storage = torch.empty((1, n, heads, 384), device='cuda', dtype=dtype)
                    t = storage[...,128:256].permute(0,2,1,3); t.copy_(x)
                else: t = x.cuda()
                tensors.append(t)
            actual = torch.empty_like(tensors[0]); original = torch.empty_like(tensors[0])
            left = baseline.ChunkedMatrixAttention(*tensors, original)
            right = candidate.ChunkedMatrixAttention(*tensors, actual)
            tolerance = .004 if dtype==torch.bfloat16 else (.0006 if dtype==torch.float16 else 1e-5)
            for repeat in range(2):
                if time.monotonic()-started > 1500: raise TimeoutError('25-minute correctness budget reached')
                for x, y in zip(tensors, cpu): x.copy_(y if repeat==0 else -y)
                left(*tensors, original); right(*tensors, actual); torch.cuda.synchronize()
                assert torch.isfinite(actual).all().item()
                torch.testing.assert_close(actual.float(), original.float(), atol=tolerance, rtol=tolerance)
                sampled = actual[:, selected_heads][:, :, queries].cpu().float()
                oracle = expected if repeat==0 else -expected
                torch.testing.assert_close(sampled, oracle, atol=tolerance, rtol=tolerance)
                row = dict(tokens=n, heads=heads, dtype=str(dtype), strided=strided, repeat=repeat,
                    status='PASS', all_output_max_abs=(actual.float()-original.float()).abs().max().item(),
                    sampled_cpu_fp64_max_abs=(sampled-oracle).abs().max().item(),
                    sampled_queries=queries, sampled_heads=selected_heads,
                    candidate_workspace_bytes=sum(t.numel()*t.element_size() for t in right.buffers)+right.scores.numel()*4)
                report['cases'].append(row); a.output.write_text(json.dumps(report, indent=2)); print(json.dumps(row), flush=True)
            del actual, original, left, right, tensors, sampled
            gc.collect(); torch.cuda.empty_cache()
        del cpu, expected
    report['status'] = 'PASS'
except BaseException as error:
    report.update(status='FAIL', error=type(error).__name__+': '+str(error)); raise
finally:
    report['wall_seconds'] = time.monotonic()-started
    a.output.write_text(json.dumps(report, indent=2))
