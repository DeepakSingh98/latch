#!/usr/bin/env bash
set -euo pipefail
: "${TINYMOE_ROOT:?Set TINYMOE_ROOT to an absolute owned output directory}"
[[ "$TINYMOE_ROOT" = /* ]] || { echo 'TINYMOE_ROOT must be absolute' >&2; exit 2; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# LATCH-V0-NEXT: the accepted exact execution rules are selected by default (prepare(rules="auto")).
# TINYMOE_RULES=none rebuilds the v0 0.8.0rc1 option set for comparison.
rules_args=()
case "${TINYMOE_RULES:-auto}" in
  auto) rules_args=(--rules auto) ;;
  none) ;;
  *) echo 'TINYMOE_RULES must be auto or none' >&2; exit 2 ;;
esac
case "${1:-}" in
  prepare)
    latch-tinymoe data download --model-dir "$TINYMOE_ROOT/model"
    latch-tinymoe data prepare --model-dir "$TINYMOE_ROOT/model" \
      --out "$TINYMOE_ROOT/data" --train-tokens 65537 --eval-tokens 8193
    ;;
  build)
    latch-tinymoe build --model "$TINYMOE_ROOT/model" \
      --out "$TINYMOE_ROOT/artifact" --cache-dir "$TINYMOE_ROOT/cache" \
      --recipe unchanged --phase plain --horizon 400 --batch-size 2 --sequence-length 512 \
      --compact-elementwise --outline-full-sums --resource-phase-local-pointers \
      --outline-pure-intervals --native-code-only --device-units 16 \
      --device-unit-inline isolate --device-unit-split 150000 --device-unit-jobs 8 \
      --static-shared-reserve-bytes 8192 --explain ${rules_args[@]+"${rules_args[@]}"}
    ;;
  train|conventional)
    backend=latch
    [[ "$1" != conventional ]] || backend="${TINYMOE_CONVENTIONAL:-compiled-grouped}"
    observe_args=()
    [[ "${TINYMOE_OBSERVE:-0}" != 1 ]] || observe_args=(--observe-at 1 32)
    latch-tinymoe train --backend "$backend" --model-dir "$TINYMOE_ROOT/model" \
      --data-dir "$TINYMOE_ROOT/data" --plain-dir "$TINYMOE_ROOT/artifact" \
      --out "$TINYMOE_ROOT/$backend" --recipe unchanged --horizon 400 --updates 32 \
      --batch-size 2 --sequence-length 512 --eval-every 16 --eval-windows 8 \
      --checkpoint-at 15 --save-final "${observe_args[@]}"
    ;;
  evaluate)
    latch-tinymoe evaluate --checkpoint "$TINYMOE_ROOT/latch/checkpoint-000032.pt" \
      --model-dir "$TINYMOE_ROOT/model" --data-dir "$TINYMOE_ROOT/data" \
      --split test --windows 8 --out "$TINYMOE_ROOT/evaluation"
    ;;
  restart)
    latch-tinymoe train --backend latch --model-dir "$TINYMOE_ROOT/model" \
      --data-dir "$TINYMOE_ROOT/data" --plain-dir "$TINYMOE_ROOT/artifact" \
      --out "$TINYMOE_ROOT/resumed" --recipe unchanged --horizon 400 --updates 32 \
      --batch-size 2 --sequence-length 512 --eval-every 16 --eval-windows 8 \
      --resume "$TINYMOE_ROOT/latch/checkpoint-000015.pt" --save-final
    ;;
  compare-restart)
    python -m megakernel_lab.tinymoe.compare_checkpoints \
      --left "$TINYMOE_ROOT/latch/checkpoint-000032.pt" \
      --right "$TINYMOE_ROOT/resumed/checkpoint-000032.pt" \
      --out "$TINYMOE_ROOT/restart-comparison"
    ;;
  screen)
    latch-tinymoe benchmark screen --model-dir "$TINYMOE_ROOT/model" \
      --data-dir "$TINYMOE_ROOT/data" --out "$TINYMOE_ROOT/control-screen" \
      --recipe unchanged --horizon 400 --batch-size 2 --sequence-length 512 \
      --updates 16 --warmup 4 --timeout-seconds 600
    ;;
  benchmark)
    latch-tinymoe benchmark paired --model-dir "$TINYMOE_ROOT/model" \
      --data-dir "$TINYMOE_ROOT/data" --plain-dir "$TINYMOE_ROOT/artifact" \
      --out "$TINYMOE_ROOT/paired" --recipe unchanged --horizon 400 \
      --batch-size 2 --sequence-length 512 --conventional "${TINYMOE_CONVENTIONAL:-compiled-grouped}" \
      --warmup 4 --updates 16 --pairs 5
    ;;
  *) echo 'Usage: quickstart.sh prepare|build|train|conventional|evaluate|restart|compare-restart|screen|benchmark' >&2; exit 2 ;;
esac
