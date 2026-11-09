#!/usr/bin/env bash
# One prescribed public sample in an already-authorized whole eight-card job.
set -euo pipefail
package="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${H3_PYTHON:?Set the verified CPython 3.11 executable}"
: "${H3_LM_ROOT:?Set the rebuilt participant InfiniLM source path}"
: "${H3_CORE_ROOT:?Set the rebuilt participant InfiniCore source path}"
: "${INFINI_ROOT:?Set the rebuilt native installation prefix}"
: "${DTK_ROOT:?Set the matching competition DTK root}"
: "${H3_MODEL_VIEW:?Create the verified read-only original model view first}"
: "${H3_FIRST_FRAME:?Set the organizer supplied uniform dog image path}"
: "${H3_RUN_DIR:?Choose a new evidence and output directory}"
test ! -e "$H3_RUN_DIR" || { echo 'Choose a fresh H3_RUN_DIR; existing evidence is preserved' >&2; exit 2; }
mkdir -p -- "$H3_RUN_DIR"
scratch="$(mktemp -d -- "$PWD/.h3-execution.XXXXXXXX")"
cleanup() {
    case "$scratch" in "$PWD"/.h3-execution.*) rm -rf -- "$scratch" ;; *) echo 'Refused cleanup outside owned scratch' >&2 ;; esac
}
trap cleanup EXIT
export TMPDIR="$scratch" XDG_CACHE_HOME="$scratch/cache" TRITON_CACHE_DIR="$scratch/cache/triton"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTORCH_NVML_BASED_CUDA_CHECK=1
export NINETOOTHED_DISABLE_FP_FUSION=1 TRITON_HIP_PROPERTIES_V2=1 TRITON_HIP_SCALAR_PROPERTIES=1
export TRITON_HIP_INCLUDE_PATH="$DTK_ROOT/include"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${H3_VENDOR_COMPILER:+$H3_VENDOR_COMPILER:}$H3_LM_ROOT/python:$H3_CORE_ROOT/python:$package:$package/runtime${PYTHONPATH:+:$PYTHONPATH}"
torch_lib="$("$H3_PYTHON" -c 'import pathlib,torch;print(pathlib.Path(torch.__file__).parent/"lib")')"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$torch_lib:$DTK_ROOT/llvm/gcvm/lib:$DTK_ROOT/cuda/cuda/lib64:$DTK_ROOT/lib:$DTK_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$H3_PYTHON" -c 'import hashlib,sys;assert hashlib.file_digest(open(sys.argv[1],"rb"),"sha256").hexdigest()=="b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54","Use the prescribed uniform first frame"' "$H3_FIRST_FRAME"
"$H3_PYTHON" -u "$package/scripts/preflight_eight_cards.py" --model-root "$H3_MODEL_VIEW" \
    --report "$H3_RUN_DIR/environment-probe.json" 2>&1 | tee "$H3_RUN_DIR/environment-probe.log"
TRITON_FORCE_HIP_ARCH="$("$H3_PYTHON" -c 'import json,sys;d=json.load(open(sys.argv[1]));a={v["arch"].split(":")[0] for v in d["devices"]};assert len(a)==1;print(a.pop())' "$H3_RUN_DIR/environment-probe.json")"
export TRITON_FORCE_HIP_ARCH
"$H3_PYTHON" -u "$package/scripts/qualify_eight_cards.py" --output-dir "$H3_RUN_DIR/components" \
    2>&1 | tee "$H3_RUN_DIR/components.log"
export H3_EIGHT_CARD_QUALIFICATION="$H3_RUN_DIR/components/eight-card-qualification.json"
"$H3_PYTHON" -u "$package/run_eightcard.py" fl2va \
    --infinilm-root "$H3_LM_ROOT" --model-path "$H3_MODEL_VIEW" --first-frame "$H3_FIRST_FRAME" \
    --text 'A dog falls asleep' --gpus 0,1,2,3,4,5,6,7 --steps 50 --num-frames 124 --seed 42 \
    --output "$H3_RUN_DIR/generated.mp4" 2>&1 | tee "$H3_RUN_DIR/generation.log"
"$H3_PYTHON" "$package/inspect_output.py" --output "$H3_RUN_DIR/generated.mp4" \
    --audit "$H3_RUN_DIR/generated.mp4.run.json" --report "$H3_RUN_DIR/media-spec.json" \
    --width 768 --height 928 --frames 124 --fps 24
