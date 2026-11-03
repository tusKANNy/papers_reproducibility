#!/bin/bash
# Build every system of the sparse benchmarks from its upstream repository, at the commit the
# paper used, with our patch where there is one (sparse/patches; each adds only the end-of-pass
# markers the timing reads, and per-query times when QUERY_TIMES_DIR is set).
#
#   SPARSE_ROOT=/path/to/sparse sparse/install.sh            # all systems
#   SPARSE_ROOT=/path/to/sparse sparse/install.sh lsp sindi  # some of them
#
# Layout under $SPARSE_ROOT: repos/<system> (sources and Rust builds), build/vsag-build-release
# (SINDI), venv-pyanns (PyANN). These are the defaults the harness reads, so env.sh needs only
# SPARSE_ROOT. Needs: git, a nightly Rust toolchain (rustup), cmake, a C++17 compiler, Python 3.12.
set -euo pipefail
ROOT="${SPARSE_ROOT:?set SPARSE_ROOT to the directory for the builds}"
PATCHES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches"
export RUSTFLAGS="-C target-cpu=native"
mkdir -p "$ROOT/repos" "$ROOT/build"

# clone <url> <commit> <dir> [patch]
clone() {
    local dir="$ROOT/repos/$3"
    [ -d "$dir/.git" ] || git clone --quiet "$1" "$dir"
    git -C "$dir" checkout --quiet "$2"
    if [ -n "${4:-}" ] && ! git -C "$dir" apply --reverse --check "$PATCHES/$4" 2>/dev/null; then
        git -C "$dir" apply "$PATCHES/$4"   # fails loudly if the patch does not apply
    fi
    echo "$3: $(git -C "$dir" rev-parse --short HEAD)${4:+ + $4}"
}

kannolo() {  # the command-line tools; the Python package is installed with pip (README)
    clone https://github.com/TusKANNy/kannolo.git v0.15.1 kannolo kannolo-v0.15.1-pass-markers.patch
    (cd "$ROOT/repos/kannolo" && cargo build --release --features cli \
        --bin hnsw_build --bin hnsw_search --bin hnsw_rerank_build_sparse --bin hnsw_rerank_search_sparse)
}

seismic() {  # Seismic and Seismic-Wave
    clone https://github.com/TusKANNy/seismic.git cc065cf seismic seismic-cc065cf-pass-markers.patch
    (cd "$ROOT/repos/seismic" && cargo build --release --bin build_inverted_index --bin perf_inverted_index)
}

lsp() {  # LSP, and the tools that prepare its input (sparse/tools/prepare_lsp_input.py)
    clone https://github.com/thefxperson/hierarchical_pruning.git fa0ede5 hierarchical_pruning
    (cd "$ROOT/repos/hierarchical_pruning" && cargo +nightly build --release)
    clone https://github.com/pisa-engine/ciff.git 0689e2a ciff
    (cd "$ROOT/repos/ciff" && cargo build --release --bin jsonl2ciff)
    clone https://github.com/JMMackenzie/enhanced-graph-bisection.git 490205b enhanced-graph-bisection
    # GAIN=approx_1 selects the approximate gain function, as DSP's and LSP's input preparation does.
    (cd "$ROOT/repos/enhanced-graph-bisection" && GAIN=approx_1 cargo build --release)
}

sindi() {  # the sparse branch of vsag, SINDI's authors' code
    clone https://github.com/Roxanne0321/vsag.git 93e034f vsag vsag-93e034f.patch
    # The four DISABLE_*_FORCE flags are passed explicitly, as SINDI's authors advise, so that
    # the SIMD selection does not depend on cmake defaults.
    (cd "$ROOT/repos/vsag" && cmake -B"$ROOT/build/vsag-build-release" -S. -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release -DNUM_BUILDING_JOBS="$(nproc)" -DENABLE_TOOLS=ON \
        -DENABLE_INTEL_MKL=OFF -DENABLE_WERROR=OFF -DDISABLE_SSE_FORCE=OFF -DDISABLE_AVX_FORCE=OFF \
        -DDISABLE_AVX2_FORCE=OFF -DDISABLE_AVX512_FORCE=OFF \
        && cmake --build "$ROOT/build/vsag-build-release" --parallel "$(nproc)")
}

pyanns() {  # its own environment: it builds against pinned numpy and pybind11
    clone https://github.com/hhy3/pyanns.git cbb5c81 pyanns
    python3.12 -m venv "$ROOT/venv-pyanns"
    "$ROOT/venv-pyanns/bin/pip" install --quiet numpy==2.5.2 pybind11==2.11.1 setuptools==84.0.0 \
        wheel==0.48.0 packaging==26.3 psutil==7.2.2
    (cd "$ROOT/repos/pyanns/python" && rm -rf build dist \
        && "$ROOT/venv-pyanns/bin/python" setup.py --quiet bdist_wheel \
        && "$ROOT/venv-pyanns/bin/pip" install --quiet --force-reinstall --no-deps dist/*.whl)
}

systems=("$@")
[ $# -gt 0 ] || systems=(kannolo seismic lsp sindi pyanns)
for s in "${systems[@]}"; do "$s"; done
echo "built under $ROOT"
