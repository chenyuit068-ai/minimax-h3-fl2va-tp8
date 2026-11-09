#!/usr/bin/env bash
# Run only inside a platform-authorized whole eight-DCU allocation.
# This script does not request resources or retry failed stages.
set -euo pipefail
stage="${1:?Use probe, qualify, or generate}"
package="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${IFAB_AFS_HOME:?Personal AFS must be mounted}"
: "${H3_PYTHON:?Use the verified absolute Python executable path}"
: "${H3_MODEL_VIEW:?Set the verified modular model-view path}"
: "${H3_LM_ROOT:?Set the participant InfiniLM source path}"
: "${H3_CORE_ROOT:?Set the participant InfiniCore source path}"
: "${H3_RUN_DIR:?Choose a fresh personal AFS evidence/output directory}"
: "${INFINI_ROOT:?Set the participant native installation prefix}"
export PYTHONPATH="$H3_LM_ROOT/python:$H3_CORE_ROOT/python:$package:$package/runtime${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NINETOOTHED_DISABLE_FP_FUSION=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p -- "$H3_RUN_DIR"
case "$stage" in
  probe)
    "$H3_PYTHON" -u "$package/scripts/preflight_eight_cards.py" \
      --model-root "$H3_MODEL_VIEW" --report "$H3_RUN_DIR/environment-probe.json" \
      2>&1 | tee "$H3_RUN_DIR/environment-probe.log"
    ;;
  qualify)
    "$H3_PYTHON" -u "$package/scripts/qualify_eight_cards.py" \
      --output-dir "$H3_RUN_DIR/components" 2>&1 | tee "$H3_RUN_DIR/components.log"
    ;;
  generate)
    : "${H3_EIGHT_CARD_QUALIFICATION:?Set the actual passed component report, not a host check}"
    : "${H3_FIRST_FRAME:?Set the organizer supplied uniform first-frame image path}"
    "$H3_PYTHON" -u "$package/run_eightcard.py" fl2va \
      --infinilm-root "$H3_LM_ROOT" --model-path "$H3_MODEL_VIEW" \
      --first-frame "$H3_FIRST_FRAME" \
      --text 'A dog falls asleep' --gpus 0,1,2,3,4,5,6,7 \
      --steps 50 --num-frames 124 --seed 42 --output "$H3_RUN_DIR/generated.mp4" \
      2>&1 | tee "$H3_RUN_DIR/generation.log"
    ;;
  *) echo 'Unknown stage; use probe, qualify, or generate' >&2; exit 2 ;;
esac
