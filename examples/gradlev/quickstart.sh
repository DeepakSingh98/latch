#!/usr/bin/env bash
# GradLev causal test-time adaptation of GPT-2's final MLP through Latch's vector_resident loop schedule.
# Stages: prepare (CPU, internet) | build (CUDA toolkit; GPU not needed) | run | conventional | benchmark | restart (one H100)
set -euo pipefail
: "${GRADLEV_ROOT:?Set GRADLEV_ROOT to an absolute, new output directory}"
[[ "$GRADLEV_ROOT" = /* ]] || { echo 'GRADLEV_ROOT must be absolute' >&2; exit 2; }
export GRADLEV_ROOT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
py="${PYTHON:-python}"
blocks="${GRADLEV_BLOCKS:-16}"                       # first N dev blocks (1,024 scored targets each)
policy="${GRADLEV_POLICY:-P1}"                       # P1 (primary) or P1b (FP32 softmax-gradient weights)
case "$policy" in
  P1) latch=latch; control_default=graph_hybrid; arms_default='graph;graph_hybrid;latch' ;;
  P1b) latch=latchgl32; control_default=graph_gl32red; arms_default='graph_gl32;graph_gl32red;latchgl32' ;;
  *) echo 'GRADLEV_POLICY must be P1 or P1b' >&2; exit 2 ;;
esac
control="${GRADLEV_CONVENTIONAL:-$control_default}"
case "${1:-}" in
  prepare)
    "$py" -B -m gradlev_example prepare --dev-windows "${GRADLEV_DEV_WINDOWS:-128}" \
      --test-windows "${GRADLEV_TEST_WINDOWS:-0}"
    ;;
  build)
    "$py" -B -m gradlev_example build --policy "$policy"
    ;;
  run)
    "$py" -B -m gradlev_example evaluate --out "$GRADLEV_ROOT/run-$latch" --split dev \
      --blocks "$blocks" --backends "$latch"
    ;;
  conventional)
    "$py" -B -m gradlev_example evaluate --out "$GRADLEV_ROOT/conventional-$control" --split dev \
      --blocks "$blocks" --backends "$control"
    ;;
  benchmark)
    arms="${GRADLEV_ARMS:-$arms_default}"
    "$py" -B -m gradlev_example benchmark --out "$GRADLEV_ROOT/benchmark-$policy" --split dev \
      --arms "$arms" --blocks "$blocks" --rounds "${GRADLEV_ROUNDS:-5}"
    ;;
  restart)
    "$py" -B -m gradlev_example restart --out "$GRADLEV_ROOT/restart"
    ;;
  *) echo 'Usage: quickstart.sh prepare|build|run|conventional|benchmark|restart' >&2; exit 2 ;;
esac
