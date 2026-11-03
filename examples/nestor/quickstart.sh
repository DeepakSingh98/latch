#!/usr/bin/env bash
# Nestor training programs with Latch: export the upstream step-0 state, then capture and build the accumulate and
# final programs of the B8 x 4096 update with one option set.
#   NESTOR_ROOT      absolute owned output directory (keep the same path on every host that reads the artifacts)
#   NESTOR_SRC       git checkout of https://github.com/Ma-Lab-Berkeley/Nestor at 9c25256fd358f4f3b0b92da175158db20fc035b3
#   PYTHON           interpreter with megakernel-lab and torch (build, smoke-build, ir-hash); default python
#   UPSTREAM_PYTHON  interpreter of the upstream Nestor environment (inventory, smoke); default $PYTHON
set -euo pipefail
: "${NESTOR_ROOT:?Set NESTOR_ROOT to an absolute owned output directory}"
[[ "$NESTOR_ROOT" = /* ]] || { echo 'NESTOR_ROOT must be absolute' >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY="${PYTHON:-python}"; UPY="${UPSTREAM_PYTHON:-$PY}"; IN="$NESTOR_ROOT/in"; B=8
need_src() { : "${NESTOR_SRC:?Set NESTOR_SRC to the upstream Nestor checkout}"; }
inventory() {  # $1 = fixture
  need_src
  "$UPY" -B "$HERE/nestor_inventory.py" --nestor-src "$NESTOR_SRC" --fixture "$1" --out "$IN"
}
case "${1:-}" in
  inventory)     # upstream environment, CPU: nestor_100m step-0 state, parameter inventory, optimizer groups
    inventory full ;;
  build)         # CPU worker with CUDA 12.8 nvcc and no visible GPU: both programs at B8
    for mode in accumulate final; do
      out="$NESTOR_ROOT/full-$mode-b$B"
      "$PY" -B "$HERE/nestor_build.py" --inventory "$IN/inventory-full.json" --init "$IN/init-full.pt" \
        --mode "$mode" --batch $B --device-unit-jobs 28 --out "$out" --cache-dir "$out/cache"
    done ;;
  smoke)         # CPU, minutes: reduced r1 fixture, capture only (checks the model, optimizer rule and capture)
    inventory r1
    for mode in accumulate final; do
      "$PY" -B "$HERE/nestor_build.py" --inventory "$IN/inventory-r1.json" --init "$IN/init-r1.pt" \
        --mode "$mode" --batch 1 --capture-only --out "$NESTOR_ROOT/smoke-r1-$mode"
    done ;;
  ir-hash)       # CPU: capture the full programs at B8 and print the captured-graph sha256 of each
    for mode in accumulate final; do
      "$PY" -B "$HERE/nestor_build.py" --inventory "$IN/inventory-full.json" --init "$IN/init-full.pt" \
        --mode "$mode" --batch $B --capture-only --ir-sha256 --out "$NESTOR_ROOT/ir-$mode-b$B"
    done ;;
  options)       # print the complete build keyword set
    "$PY" -B "$HERE/nestor_build.py" --print-options ;;
  *) echo 'Usage: quickstart.sh inventory|build|smoke|ir-hash|options' >&2; exit 2 ;;
esac
