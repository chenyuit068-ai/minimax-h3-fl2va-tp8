#!/usr/bin/env bash
# Runs independently of the model entry. No inference result is retained/reused.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
package="$(cd -- "$here/../.." && pwd)"
destination="${1:?Pass a fresh evidence directory}"
mode="${2:-selected}"
case "$mode" in selected|all) ;; *) echo 'Mode must be selected or all' >&2; exit 2;; esac
test ! -e "$destination" || { echo 'Use a new evidence directory' >&2; exit 2; }
mkdir -p "$destination"
export NINETOOTHED_DISABLE_FP_FUSION=1
export PYTHONPATH="$here/v4-runtime:$here${PYTHONPATH:+:$PYTHONPATH}"
python - "$here" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
for row in json.loads((root/'source-manifest.json').read_text())['files']:
    assert hashlib.sha256((root/row['file']).read_bytes()).hexdigest()==row['sha256'],row['file']
print('EXPERIMENT_SOURCE_CHECKSUM_PASS')
PY
python "$here/test_block_softmax_target_20261023.py" --output "$destination/block-softmax.json"
backend="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_backend"])' "$package/attention-backend-manifest.json")"
case "$backend" in
    split)
        python "$here/test_split_attention_target_20261023.py" --runtime "$here/v4-runtime" --output "$destination/split-attention.json"
        ;;
    pv_fused)
        python "$here/test_pv_fused_unit_20261023.py" --runtime "$here/v4-runtime" --output "$destination/pv-fused-unit.json"
        python "$here/test_pv_fused_attention_20261023.py" --runtime "$here/v4-runtime" --output "$destination/pv-fused-attention.json"
        ;;
    *) echo 'Unsupported selected backend in this release' >&2; exit 2;;
esac
# These scripts insert the current production runtime themselves, so their
# native bridge/tiny-model checks qualify the delivered selected backend.
python "$package/tests/test_native_matrix_attention.py" --output "$destination/native-dispatch.json"
python "$package/tests/test_streaming_tiny_h3.py" --matrix --output "$destination/tiny-h3.json"
python "$package/tests/test_streaming_tiny_qwen.py" --matrix --output "$destination/tiny-qwen.json"
if test "$mode" = all; then
    if test "$backend" != split; then
        python "$here/test_split_attention_target_20261023.py" --runtime "$here/v4-runtime" --output "$destination/split-attention.json"
    fi
    python "$here/test_online_attention_target_20261023.py" --runtime "$here/v4-runtime" --output "$destination/online-attention.json"
    python "$here/benchmark_attention_candidates_20261023.py" --runtime "$here/v4-runtime" --output "$destination/attention-paired.json"
    if test "$backend" != pv_fused; then
        python "$here/test_pv_fused_unit_20261023.py" --runtime "$here/v4-runtime" --output "$destination/pv-fused-unit.json"
        python "$here/test_pv_fused_attention_20261023.py" --runtime "$here/v4-runtime" --output "$destination/pv-fused-attention.json"
    fi
    python "$here/benchmark_pv_fused_attention_20261023.py" --runtime "$here/v4-runtime" --output "$destination/pv-fused-paired.json"
fi
