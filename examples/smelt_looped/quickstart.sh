#!/usr/bin/env bash
# Provenance: adapted from the private Latch repository (branch codex/smelt-latch) at commit 2c9ddb2e3d51
# (branch codex/smelt-latch), original path examples/smelt_looped/quickstart.sh (blob 554e896bfd23,
# sha256 f39072c7c93c...; last changed at 2b1084643).
# Changed: uses the INSTALLED megakernel-lab package (no repository src/ on PYTHONPATH); PYTHONPATH is this
# directory only; the build stage runs the local smelt_build.py instead of research.smelt_latch.build; SMELT_LOOP
# is validated. Stages, B8 x S512, the 32 updates, data sizes and build flags are the donor's.
#
# Looped (SMELT-style) TinyMoE training example: generated Latch update vs conventional execution of the SAME model.
#   SMELT_ROOT   absolute owned output directory (same path on the CPU build host and the H100)
#   SMELT_MODEL  hash-pinned TinyMoE snapshot (default $SMELT_ROOT/model, written by `prepare`)
#   SMELT_DATA   `latch-tinymoe data prepare` token directory (default $SMELT_ROOT/data)
#   SMELT_LOOP   FIRST,SPAN,PASSES (default 3,6,2: 1-based layers 4-9 visited twice, residual updates x 1/2;
#                0,1,1 = one pass, i.e. the unlooped recipe-Q program). Use a separate SMELT_ROOT per loop.
#   SMELT_ARTIFACT  built artifact directory (default $SMELT_ROOT/artifact)
set -euo pipefail
: "${SMELT_ROOT:?Set SMELT_ROOT to an absolute owned output directory}"
[[ "$SMELT_ROOT" = /* ]] || { echo 'SMELT_ROOT must be absolute' >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MODEL="${SMELT_MODEL:-$SMELT_ROOT/model}"; DATA="${SMELT_DATA:-$SMELT_ROOT/data}"; LOOP="${SMELT_LOOP:-3,6,2}"
[[ "$LOOP" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]] || { echo 'SMELT_LOOP must be FIRST,SPAN,PASSES, e.g. 3,6,2 or 0,1,1' >&2; exit 2; }
B=8; S=512; PY="${PYTHON:-python}"; ART="${SMELT_ARTIFACT:-$SMELT_ROOT/artifact}"
common=(--model-dir "$MODEL" --data-dir "$DATA" --loop "$LOOP" --batch-size $B --sequence-length $S --updates 32)
case "${1:-}" in
  prepare)       # CPU, internet: pinned public model/tokenizer and a small WikiText-103 token split
    latch-tinymoe data download --model-dir "$MODEL"
    latch-tinymoe data prepare --model-dir "$MODEL" --out "$DATA" --train-tokens 262145 --eval-tokens 8193 ;;
  build)         # CPU worker with the CUDA toolkit (no visible GPU): capture + normal-API Latch build (rules="auto")
    "$PY" -B "$HERE/smelt_build.py" --model "$MODEL" --loop "$LOOP" --batch-size $B --sequence-length $S \
      --out "$ART" --cache-dir "$SMELT_ROOT/cache" --device-unit-jobs 8 ;;
  train)         # H100: 32 generated updates, checkpoint, dev NLL
    "$PY" -B "$HERE/example.py" train --backend latch --artifact "$ART" "${common[@]}" --out "$SMELT_ROOT/latch" ;;
  conventional)  # H100: same model/data/optimizer, compiled + CUDA-graph conventional execution
    "$PY" -B "$HERE/example.py" train --backend conventional "${common[@]}" --out "$SMELT_ROOT/conventional" ;;
  restart)       # H100: stop after update 15 in one process, resume to 32 in a new process
    "$PY" -B "$HERE/example.py" train --backend latch --artifact "$ART" "${common[@]}" --save-at 15 --out "$SMELT_ROOT/first"
    "$PY" -B "$HERE/example.py" train --backend latch --artifact "$ART" "${common[@]}" \
      --resume "$SMELT_ROOT/first/checkpoint-000015.pt" --out "$SMELT_ROOT/resumed" ;;
  compare-restart)
    "$PY" -B "$HERE/example.py" compare "$SMELT_ROOT/latch/checkpoint-000032.pt" "$SMELT_ROOT/resumed/checkpoint-000032.pt" ;;
  *) echo 'Usage: quickstart.sh prepare|build|train|conventional|restart|compare-restart' >&2; exit 2 ;;
esac
