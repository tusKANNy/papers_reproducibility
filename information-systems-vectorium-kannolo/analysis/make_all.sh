#!/usr/bin/env bash
# Regenerate the paper's Section 6 tables, figures and supplementary configuration lists from the
# rows in results/:  analysis/make_all.sh <out-dir>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
OUT="${1:?usage: make_all.sh <out-dir>}"; mkdir -p "$OUT"
export SOURCE_DATE_EPOCH=0 PYTHONHASHSEED=0   # byte-stable PDFs
S="$REPO/selections"; mkdir -p "$S"
while read -r kind name budget; do
  python3 "$HERE/select_configs.py" --results "$REPO/results/$kind/$name.jsonl" --budget-s "$budget" \
    --series "$kind" --out "$S/${kind}_$name.json"
done <<'LIST'
dense agnews 300
dense msmarco-qwen 2400
dense dpr 5400
dense laion 300
sparse msmarco-splade 5400
sparse nq 1800
sparse msmarco-esplade 6000
LIST
cd "$HERE"
python3 combined.py --series dense --out-dir "$OUT" --table-dir "$OUT" --selection \
  agnews="$S/dense_agnews.json" msmarco-qwen="$S/dense_msmarco-qwen.json" \
  dpr="$S/dense_dpr.json" laion="$S/dense_laion.json"
python3 combined.py --series sparse --out-dir "$OUT" --table-dir "$OUT" --selection \
  msmarco-splade="$S/sparse_msmarco-splade.json" nq="$S/sparse_nq.json" \
  msmarco-esplade="$S/sparse_msmarco-esplade.json"
python3 tail_table.py "$OUT/tail_latency.tex"
python3 budget_sensitivity.py --results "$REPO/results/dense/dpr.jsonl" --budget-s 5400 > "$OUT/budget_sensitivity_dpr.md"
