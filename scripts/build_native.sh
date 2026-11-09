#!/usr/bin/env bash
set -euo pipefail
package="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo="${1:?Pass the participant InfiniLM repository, not a protected baseline checkout}"
test -n "${INFINI_ROOT:-}" || { echo 'Activate the platform environment first'; exit 2; }
python "$package/scripts/apply_native_patch.py" --repo "$repo"
python "$package/scripts/apply_conditioner_patch.py" --repo "$repo"
cd "$repo"
export PYTHONPATH="$repo/python${PYTHONPATH:+:$PYTHONPATH}"
# All binding translation units must use the same pybind11 ABI as InfiniCore.
xmake build -j2 _infinilm
xmake install _infinilm
python -c 'import torch; from infinilm.lib import _infinilm; assert hasattr(_infinilm,"_h3_set_nine_callback"); print("NATIVE_BRIDGE_BUILD_PASS")'
