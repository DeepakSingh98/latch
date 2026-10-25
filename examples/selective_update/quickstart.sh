#!/usr/bin/env bash
# TinyMoE selective-update (selective cadence, arm A4) complete-training example. See README.md.
set -euo pipefail
: "${SELECTIVE_ROOT:?Set SELECTIVE_ROOT to an absolute owned output directory}"
[[ "$SELECTIVE_ROOT" = /* ]] || { echo 'SELECTIVE_ROOT must be absolute' >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R="$SELECTIVE_ROOT"
PY="${PYTHON:-python}"
# Start state: the public pretrained snapshot (default) or SELECTIVE_INIT=fresh (random init, seed SELECTIVE_SEED).
# Schedule position u0 (study legacy schedule): default 32768 (the study's E point, LR ~ the 3e-4 peak) from the
# pretrained snapshot, 0 (start of warmup, as the donor's fresh state) from a random init.
init_args=()
case "${SELECTIVE_INIT:-pretrained}" in
  pretrained) default_u0=32768 ;;
  fresh) init_args=(--init "$R/init/init-s${SELECTIVE_SEED:-1743}.pt"); default_u0=0 ;;
  *) echo 'SELECTIVE_INIT must be pretrained or fresh' >&2; exit 2 ;;
esac
START=(--start-update "${SELECTIVE_START_UPDATE:-$default_u0}" --order-seed "${SELECTIVE_ORDER_SEED:-1729}")
# The qualified LATCH-SELECTIVE-10 C18 build options (both phases).
BUILD=(--slow-group experts --device-unit-jobs 16 --update-schedule streamed_prep_ustate_late
       --segmented-products flat+tma+rotate --norm-fusion rows --attention-rules streamed
       --gradient-outputs consumed
       --accumulation-rules castview+fsgather+fswarp+wsoftn+fsdeep+rorder+cmulti+rowchain+fsmix+fsfence+dcast)
# 64 microbatches = 4 complete 16-microbatch windows; evaluations at 0, 32, 64 (16 dev windows each).
RUN=(--accumulate "$R/accumulate" --apply "$R/apply" --model-dir "$R/model" --data-dir "$R/windows"
     --microbatches 64 --eval-every 32 --eval-windows 16 --final-eval-windows 16
     --lr-scale "${SELECTIVE_LR_SCALE:-1.0}")
case "${1:-}" in
  prepare)
    latch-tinymoe data download --model-dir "$R/model"
    latch-tinymoe data prepare --model-dir "$R/model" \
      --out "$R/data" --train-tokens 262145 --eval-tokens 32769
    [[ -e "$R/windows" ]] || "$PY" -m megakernel_lab.tinymoe.learning quick-windows --source "$R/data" --out "$R/windows"
    if [[ "${SELECTIVE_INIT:-pretrained}" == fresh && ! -e "$R/init" ]]; then
      CUDA_VISIBLE_DEVICES="" "$PY" -m megakernel_lab.tinymoe.learning init-state \
        --model-dir "$R/model" --seed "${SELECTIVE_SEED:-1743}" --out "$R/init"
    fi
    ;;
  build)
    # Capture/build refuses to run with a visible GPU: build on a CPU-only process (CUDA toolkit required).
    for phase in accumulate apply; do
      CUDA_VISIBLE_DEVICES="" "$PY" "$HERE/selective_build.py" --model "$R/model" \
        --out "$R/$phase" --cache-dir "$R/cache" --phase "$phase" "${BUILD[@]}"
    done
    ;;
  train)
    kernel_args=()
    [[ "${SELECTIVE_KERNELS:-0}" != 1 ]] || kernel_args=(--kernels)
    "$PY" "$HERE/selective_train.py" "${RUN[@]}" "${START[@]}" ${init_args[@]+"${init_args[@]}"} \
      --save-at 24 --save-final --out "$R/train" ${kernel_args[@]+"${kernel_args[@]}"}
    ;;
  conventional)
    native_args=()
    [[ ! -e "$R/train/final-parameters.pt" ]] || native_args=(--native-run "$R/train")
    "$PY" "$HERE/selective_conventional.py" --model-dir "$R/model" --data-dir "$R/windows" \
      --microbatches 64 --warm 16 "${START[@]}" ${init_args[@]+"${init_args[@]}"} \
      --out "$R/conventional" ${native_args[@]+"${native_args[@]}"}
    ;;
  restart)
    "$PY" "$HERE/selective_train.py" "${RUN[@]}" "${START[@]}" ${init_args[@]+"${init_args[@]}"} \
      --resume "$R/train/checkpoint.pt" --out "$R/resumed"
    ;;
  compare-restart)
    "$PY" "$HERE/selective_compare_restart.py" --straight "$R/train" --resumed "$R/resumed" \
      --out "$R/restart-comparison"
    ;;
  *) echo 'Usage: quickstart.sh prepare|build|train|conventional|restart|compare-restart' >&2; exit 2 ;;
esac
