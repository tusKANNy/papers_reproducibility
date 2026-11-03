# Vectorium and kANNolo: benchmarks

Code, parameter grids and measured results behind Section 6 of *Modular and Efficient Approximate Nearest Neighbors Search with Vectorium and kANNolo*.
Every table, figure and configuration list of that
section is regenerated from the rows in `results/` by one script; the benchmarks themselves can be
re-run from scratch with the drivers in `dense/` and `sparse/`.
All commands below run from this folder.

```
dense/      VIBE harness: our two modules, the grid of every system, the patch to VIBE, the driver
sparse/     sparse harness: driver, one module per system, the grid of every system, patches
analysis/   selection, tables, figures, supplementary grid tables
results/    the measured rows the paper reports (dense/, sparse/, tail/)
envs/       Python environments
```

## Regenerate the paper's tables and figures

```bash
pip install matplotlib seaborn numpy pyyaml
analysis/make_all.sh out/
```

writes `results_{dense,sparse}.tex` (Tables B.1 and B.2 before compaction), `configs_{dense,sparse}.tex`
(supplementary material), `fig_frontier_*.pdf` and `fig_tradeoff_*.pdf` (Figures 6-9) and
`tail_latency.tex` (Table B.3). It also writes the budget sensitivity of Appendix A.1 and the
selections in `selections/`. `analysis/grids_table.py` writes the grid tables of the supplementary material.

**Selection rule** (`analysis/select_configs.py`): a row counts if its index was placed on the search
node (or, for SymphonyQG on DPR, which fits on no single node, interleaved) and its build finished
within the collection's budget B. Tables report, per system and target t in {90, 95, 99}, the fastest
configuration with Accuracy@10 >= t; figures draw the fastest configuration per integer accuracy
interval.

## Library versions

kANNolo 0.15.1 with Vectorium 0.11.1, Rust toolchain `nightly-2026-06-19` (pinned in kANNolo's
`rust-toolchain.toml`). Build for the machine; a generic wheel is several times slower:

```bash
RUSTFLAGS="-C target-cpu=native" pip install "kannolo @ git+https://github.com/TusKANNy/kannolo.git@v0.15.1"
```

## Dense benchmarks (VIBE)

Collections: AgNews (B = 300 s), MsMarco-Qwen (2400 s), DPR (5400 s), LAION (300 s), as shipped by
VIBE, with their ground truth.

```bash
conda env create -f envs/vibe-conda.yml && conda activate vibe   # then install kANNolo as above
git clone https://github.com/vector-index-bench/vibe "$VIBE" && git -C "$VIBE" checkout 1b60239
git -C "$VIBE" apply "$PWD/dense/vibe/patches/vibe_1b60239.patch"
cp dense/vibe/patches/placement.py "$VIBE/vibe/placement.py"
VIBE=$VIBE KANNOLO_BENCH_ROOT=$DATA dense/vibe/install.sh   # $DATA: where datasets/ and logs/ live

export KANNOLO_BENCH_VIBE=$VIBE KANNOLO_BENCH_ROOT=$DATA     # or pass --vibe / --root (--vibe-root / --data-root)
export MALLOC_ARENA_MAX=8 MALLOC_MMAP_THRESHOLD_=268435456
python dense/run_dense.py --dataset dpr-jina-768-normalized --budget-s 5400
python dense/collect_results.py --dataset dpr-jina-768-normalized --out results/dense/dpr.jsonl
python analysis/select_configs.py --results results/dense/dpr.jsonl --budget-s 5400 --out selections/dense_dpr.json
python dense/run_ext8.py --dataset dpr-jina-768-normalized --budget-s 5400 --selection selections/dense_dpr.json
python dense/collect_results.py --dataset dpr-jina-768-normalized --out results/dense/dpr.jsonl
```

The patch to VIBE changes only how indexes are built and placed; the timed search loop is
unchanged. Construction runs on all cores, and the index is then migrated to the search node
(`placement.py`). Search runs on one core, QPS is recorded per pass, and the same allocator
settings apply to every system. The grids are
VIBE's, plus each library's documented default where VIBE's grid omits it (comments in each
`config.yml`). `run_dense.py` builds every system's grid cheapest first. A build over B is excluded, and every
larger build of the same system is skipped without being built (`dense/dominance.py`).

## Sparse benchmarks

Collections: MsMarco-SPLADE-v3 (B = 5400 s), NQ with SPLADE-CoCondenser (1800 s), MsMarco with
Efficient-SPLADE (6000 s). The files used in the paper are on Hugging Face
([collection](https://huggingface.co/collections/tuskanny/sparse-datasets-6ae76dbcf8326d3aaadd0263)):
[`tuskanny/sparse-msmarco-splade-v3`](https://huggingface.co/datasets/tuskanny/sparse-msmarco-splade-v3),
[`tuskanny/sparse-nq-splade-cocondenser`](https://huggingface.co/datasets/tuskanny/sparse-nq-splade-cocondenser) and
[`tuskanny/sparse-msmarco-efficient-splade`](https://huggingface.co/datasets/tuskanny/sparse-msmarco-efficient-splade).
Each holds `documents.bin`, `queries.bin` (Seismic binary format) and `groundtruth.tsv`;
the dataset cards describe the formats.

```bash
hf download tuskanny/sparse-nq-splade-cocondenser --repo-type dataset --local-dir <nq-dir> \
  --include documents.bin queries.bin groundtruth.tsv
```

Every competitor is built from its upstream repository at the commit the paper used, with our
patch where there is one. `sparse/install.sh` clones, patches and builds all of them under
`$SPARSE_ROOT`:

| System | Repository | Commit | Patch |
|---|---|---|---|
| kANNolo (command-line tools) | [TusKANNy/kannolo](https://github.com/TusKANNy/kannolo) | `v0.15.1` | `kannolo-v0.15.1-pass-markers.patch` |
| Seismic, Seismic-Wave | [TusKANNy/seismic](https://github.com/TusKANNy/seismic) | `cc065cf` | `seismic-cc065cf-pass-markers.patch` |
| LSP | [thefxperson/hierarchical_pruning](https://github.com/thefxperson/hierarchical_pruning) | `fa0ede5` | |
| LSP input: CIFF conversion | [pisa-engine/ciff](https://github.com/pisa-engine/ciff) | `0689e2a` | |
| LSP input: graph bisection | [JMMackenzie/enhanced-graph-bisection](https://github.com/JMMackenzie/enhanced-graph-bisection) | `490205b` | |
| SINDI | [Roxanne0321/vsag](https://github.com/Roxanne0321/vsag) (branch `sparse`) | `93e034f` | `vsag-93e034f.patch` |
| PyANN | [hhy3/pyanns](https://github.com/hhy3/pyanns) | `cbb5c81` | |

The patches (`sparse/patches`) add the end-of-pass markers the timing reads and, when
`QUERY_TIMES_DIR` is set, per-query times. The SINDI patch also fixes the header that
`generate_gt` writes, and it pins vsag's ThreadPool dependency to the commit the paper's build
used (vsag otherwise downloads it from a moving `master` branch). The scripts that
convert the collections into LSP's and SINDI's input formats are in `sparse/tools`. The build needs
git, a nightly Rust toolchain, cmake, a C++17 compiler and Python 3.12.

```bash
pip install -r envs/sparse-requirements.txt          # then kANNolo as above
SPARSE_ROOT=/path/to/sparse sparse/install.sh         # or: sparse/install.sh lsp sindi
cp sparse/env.example.sh env.sh                       # set SPARSE_ROOT
set -a; . env.sh; set +a
python sparse/run_sparse.py --sparse-root $SPARSE_ROOT --collection nq --data <nq-dir> --out results/sparse/nq.jsonl
```

Every system built on Vectorium stores the collection with the blocked-sparse encoder. kANNolo
(both configurations) and PyANN are tuned in-process over their grid, and the three fastest
configurations at each accuracy target are then timed from the command line on the same index
files. Seismic, SeismicWave, LSP and SINDI are built and searched from their command-line tools.
Every reported sparse number is a command-line timing: one search process over the whole query
set on one core, timed between the tool's own end-of-pass markers, best of the two passes after
the first. The grids are in `sparse/grids.py`.

`SPARSE_SMOKE=1` (sparse) and `DENSE_SMOKE=1` (dense) run only the first build of each grid on
three search points, to check an installation.

## Machine

Two Intel Xeon Silver 4314 sockets (2 x 16 cores, 64 hardware threads), two NUMA nodes, 503 GiB RAM.
Construction uses 64 threads; search runs on core 0 of node 0.
