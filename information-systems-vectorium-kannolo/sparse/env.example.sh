#!/usr/bin/env bash
# Environment for sparse/run_sparse.py. Copy, set the paths, then `set -a; . env.sh; set +a`.
# Every system is built from source at the commit below, with the patch in sparse/patches applied
# (each adds only the end-of-pass marker the timing reads, and per-query times when
# QUERY_TIMES_DIR is set). Build with RUSTFLAGS="-C target-cpu=native" where it is a Rust crate.

SPARSE_ROOT=/path/to/sparse          # source trees, builds and caches (keep off $HOME)

# kANNolo v0.15.1 (vectorium 0.11.1) + kannolo-v0.15.0-pass-markers.patch:
#   cargo build --release --features cli --bin hnsw_search --bin hnsw_rerank_search_sparse
# and the Python package in the driver's environment, built for the machine (envs/sparse-requirements.txt).
KANNOLO_BS_BIN=$SPARSE_ROOT/repos/kannolo/target/release

# Seismic 0.6.0 (commit cc065cf) + seismic-cc065cf-pass-markers.patch: cargo build --release
SEISMIC_BS_REPO=$SPARSE_ROOT/repos/seismic

# LSP (github.com/thefxperson/hierarchical_pruning, commit fa0ede5): cargo build --release.
# Its input is prepared once per collection with replicate-dsp (commit 804d24e) +
# replicate-dsp-804d24e.patch; run_sparse.py does this when SPARSE_CIFF_PREP is unset.
LSP_REPO=$SPARSE_ROOT/repos/hierarchical_pruning
REPLICATE_DSP=$SPARSE_ROOT/repos/replicate-dsp

# SINDI through replicate-sindi-poc (commit 4d22e55) + replicate-sindi-poc-4d22e55.patch,
# built with its build_poc_vsag.sh into SINDI_POC_BUILD.
SINDI_POC_REPO=$SPARSE_ROOT/repos/replicate-sindi-poc
SINDI_POC_BUILD=$SPARSE_ROOT/build/vsag-build-release

# PyANN (commit cbb5c81) has its own Python environment (it pins numpy and pybind11 versions).
PYANNS_PYTHON=$SPARSE_ROOT/venv-pyanns/bin/python
PYANNS_CSR_CACHE=$SPARSE_ROOT/csr

# Search core and build cores (placement arm D: build on all cores, search on one).
VIBE_SEARCH_CPU=0
VIBE_SEARCH_NODE=0
