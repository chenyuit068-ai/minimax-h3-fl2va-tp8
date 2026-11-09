#!/usr/bin/env bash
# Experimental: separate participant source copies and a separate INFINI_ROOT.
set -euo pipefail
package="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
core="${1:?Pass the separate participant InfiniCore source copy}"
lm="${2:?Pass the separate participant InfiniLM source copy}"
jobs="${H3_BUILD_JOBS:-2}"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo 'H3_BUILD_JOBS must be a positive integer' >&2; exit 2; }
test -n "${INFINI_ROOT:-}" || { echo 'Set a separate INFINI_ROOT before building'; exit 2; }
if test ! -e "$lm/csrc/pybind11/bindings.cc.h3-streaming-reference"; then
    python "$package/scripts/apply_native_patch.py" --repo "$lm"
fi
python "$package/scripts/apply_streaming_patch.py" --core "$core" --lm "$lm"
python "$package/scripts/apply_conditioner_patch.py" --repo "$lm"
if test -f "$package/platform/InfiniCore-op-cache-teardown.patch"; then
    python "$package/scripts/apply_op_cache_teardown_patch.py" --core "$core"
fi
cd "$core"
# Installing the C++ API alone does not populate a new prefix with the
# foundation headers/libraries needed by InfiniLM (for example infiniop.h).
for dependency in infinirt infiniop infiniccl; do
    xmake build -j"$jobs" "$dependency"
    xmake install "$dependency"
done
xmake build -j"$jobs" infinicore_cpp_api
xmake install infinicore_cpp_api
xmake build -j"$jobs" _infinicore
xmake install _infinicore
cd "$lm"
xmake build -j"$jobs" _infinilm
xmake install _infinilm
export PYTHONPATH="$lm/python:$core/python${PYTHONPATH:+:$PYTHONPATH}"
if test "${H3_BUILD_DEVICE_IMPORT_CHECK:-1}" = 1; then
    python -c 'import torch; from infinilm.lib import _infinilm as n; assert n._h3_streaming_storage_available(); print("STREAMING_BUILD_COMPLETE_FULL_MODEL_NOT_VALIDATED")'
else
    echo STREAMING_BINARIES_BUILT_DEVICE_IMPORT_MUST_BE_TESTED_ON_ALLOCATED_NODE
fi
