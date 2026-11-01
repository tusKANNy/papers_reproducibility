#!/bin/bash
# Prepare a VIBE checkout for the dense benchmarks:
#   git clone https://github.com/vector-index-bench/vibe "$VIBE" && git -C "$VIBE" checkout 1b60239
#   git -C "$VIBE" apply "$PWD/dense/vibe/patches/vibe_1b60239.patch"
#   cp dense/vibe/patches/placement.py "$VIBE/vibe/placement.py"
#   VIBE=... KANNOLO_BENCH_ROOT=... dense/vibe/install.sh
# Our two modules are linked into VIBE's package (config.yml names them by dotted path); the grids
# stay in dense/vibe/definitions and are passed with --definitions by run_dense.py.
set -euo pipefail
VIBE="${VIBE:?set VIBE to the VIBE checkout}"
ROOT="${KANNOLO_BENCH_ROOT:?set KANNOLO_BENCH_ROOT to the data directory}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -d "$VIBE/vibe/algorithms" ] || { echo "no VIBE checkout at $VIBE" >&2; exit 1; }
for mod in "$HERE"/modules/*/; do
    ln -sfnT "$mod" "$VIBE/vibe/algorithms/$(basename "$mod")"
done
mkdir -p "$ROOT/datasets"
ln -sfnT "$ROOT/datasets" "$VIBE/data"   # VIBE reads datasets from ./data
echo "linked modules and $VIBE/data -> $ROOT/datasets"
