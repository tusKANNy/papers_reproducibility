#!/usr/bin/env bash
# Environment for sparse/run_sparse.py. Copy, set SPARSE_ROOT, then `set -a; . env.sh; set +a`.
# sparse/install.sh builds every system under SPARSE_ROOT in the layout below, which is also the
# default the harness reads; the other variables are needed only for builds kept elsewhere.

SPARSE_ROOT=/path/to/sparse          # source trees, builds and caches

KANNOLO_BS_BIN=$SPARSE_ROOT/repos/kannolo/target/release
SEISMIC_BS_REPO=$SPARSE_ROOT/repos/seismic
LSP_REPO=$SPARSE_ROOT/repos/hierarchical_pruning
JSONL2CIFF=$SPARSE_ROOT/repos/ciff/target/release/jsonl2ciff
CREATE_RGB=$SPARSE_ROOT/repos/enhanced-graph-bisection/target/release/create-rgb
SINDI_BUILD=$SPARSE_ROOT/build/vsag-build-release
PYANNS_PYTHON=$SPARSE_ROOT/venv-pyanns/bin/python
PYANNS_CSR_CACHE=$SPARSE_ROOT/csr

# Search on one core of one NUMA node, build on all cores.
VIBE_SEARCH_CPU=0
VIBE_SEARCH_NODE=0
