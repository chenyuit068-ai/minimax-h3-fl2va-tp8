#!/usr/bin/env bash
set -euo pipefail
package="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo="${1:?Pass the patched participant InfiniLM directory}"
results="${2:?Pass a new results directory}"
test ! -e "$results" || { echo 'Use a new results directory to retain earlier evidence'; exit 2; }
mkdir -p "$results"
export PYTHONPATH="$repo/python${PYTHONPATH:+:$PYTHONPATH}"
export NINETOOTHED_DISABLE_FP_FUSION=1
ulimit -c 0
python -u "$package/tests/test_native_dispatch.py" --output "$results/native-dispatch.json" 2>&1 | tee "$results/native-dispatch.log"
blocked=0
python -u "$package/tests/test_attention.py" --native --output "$results/attention.json" 2>&1 | tee "$results/attention.log"
python -u "$package/tests/test_tiny_h3.py" --output "$results/tiny-h3.json" 2>&1 | tee "$results/tiny-h3.log"
if python -c 'import torch; raise SystemExit(0 if torch.version.hip else 1)'; then
    python -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps(dict(status="BLOCKED",reason="Full official FL2VA generation and model residency are not validated",full_fl2va_tested=False),indent=2))' "$results/model-blocker.json"
    blocked=2
fi
python -u "$package/tests/benchmark_native_dispatch.py" --output "$results/native-benchmark.json" 2>&1 | tee "$results/native-benchmark.log"
python -u "$package/tests/test_host_tools.py" --comparator "$repo/examples/minimax_h3_compare.py" --output "$results/host-tools.json" 2>&1 | tee "$results/host-tools.log"
echo COMPONENT_CHECKS_COMPLETE_FULL_FL2VA_NOT_TESTED
exit "$blocked"
