#!/usr/bin/env python3
"""Prepare LSP's input for one collection, timing each step and writing the timings to a sidecar.

The steps are those of DSP's `prepare_data.sh` (LSP takes the same input). The build time of an
index includes the quantization and the graph bisection, but not the format conversions:

    seismic -> JSONL              conversion          NOT charged
    jsonl2ciff (conversion)       conversion          NOT charged
    jsonl2ciff (quantization)     preparation         CHARGED
    create-rgb graph bisection    preparation         CHARGED
    queries -> PISA               conversion          NOT charged
    extract ids                   conversion          NOT charged

A single wall-clock time for the whole script can therefore be neither charged nor excluded.
This script runs the same steps, in the same order, with the same binaries, and records the cost
of each.

**Everything here is shared across a collection's builds.** The index parameters only affect
the final `ciff2bmp`, so one preparation serves every build. The charged parts are measured once
and charged in full to each build; `prep_timings.json` records the measurement.

**Separating the quantization from the conversion.** `jsonl2ciff` does both in one binary, so
the quantization cost is obtained by subtraction: run it without `--quantize`, then with. That
difference is reliable only if the two runs see the same machine state, and they cannot. The
second run reads a JSONL that the first just loaded into the page cache, which biases the
difference downward. A quantized CIFF also *writes less* than an unquantized one, which biases
it downward again. The unquantized run is therefore done **twice, bracketing the quantized one**.
If the two brackets agree, the environment was stable and the subtraction is valid. If they do
not, the sidecar says so and the number must not be used. The fallback is to separate the step
physically, which needs a patch.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_REPOS = Path(os.environ.get("SPARSE_ROOT", "")) / "repos"  # built there by sparse/install.sh
JSONL2CIFF = Path(os.environ.get("JSONL2CIFF", _REPOS / "ciff" / "target" / "release" / "jsonl2ciff"))
CREATE_RGB = Path(os.environ.get(
    "CREATE_RGB", _REPOS / "enhanced-graph-bisection" / "target" / "release" / "create-rgb"))
EXTRACT_IDS = HERE / "extract_jsonl_ids.py"
CONV_QUERIES = HERE / "convert_queries_to_pisa.py"


def run(label, argv, log_dir):
    log = log_dir / f"{label}.log"
    t0 = time.perf_counter()
    with log.open("wb") as fh:
        rc = subprocess.call([str(a) for a in argv], stdout=fh, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    print(f"  {label}: {dt:.1f}s rc={rc}", flush=True)
    if rc != 0:
        raise SystemExit(f"{label} exited rc={rc}; see {log}")
    return dt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs-jsonl", required=True, type=Path)
    p.add_argument("--queries-jsonl", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--scale", type=int, default=1000,
                   help="PISA query weight scale. Measured on NQ, it does not affect fidelity, "
                        "because DSP renormalizes every query to MAX_TERM_WEIGHT=32. It sets "
                        "how many tiny terms are dropped and the size of queries.pisa.")
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    sidecar = out / "prep_timings.json"
    if sidecar.exists():
        print(f"{sidecar} exists; preparation already measured. Delete it to re-measure.")
        return 0

    for b in (JSONL2CIFF, CREATE_RGB):
        if not b.is_file():
            raise SystemExit(f"not built: {b}. Run sparse/install.sh first.")

    ciff = out / "docs.ciff"
    ciff_probe = out / "docs_noquant_probe.ciff"
    reordered = out / "docs_reordered.ciff"
    pisa = out / "queries.pisa"

    print("measuring the conversion/quantization split (unquantized, quantized, unquantized):")
    t_noquant_a = run("ciff-noquant-a",
                      [JSONL2CIFF, "--input", args.docs_jsonl, "--output", ciff_probe], out)
    t_quant = run("ciff-quantized",
                  [JSONL2CIFF, "--input", args.docs_jsonl, "--output", ciff, "--quantize"], out)
    ciff_probe.unlink(missing_ok=True)
    t_noquant_b = run("ciff-noquant-b",
                      [JSONL2CIFF, "--input", args.docs_jsonl, "--output", ciff_probe], out)
    ciff_probe.unlink(missing_ok=True)

    bracket_spread = abs(t_noquant_a - t_noquant_b) / max(t_noquant_a, t_noquant_b)
    t_noquant = (t_noquant_a + t_noquant_b) / 2
    quantization_s = t_quant - t_noquant
    # The brackets must agree for the subtraction to be about quantization rather than about
    # machine state. 5 % is generous.
    subtraction_valid = bracket_spread <= 0.05 and quantization_s > 0

    print("graph bisection (CHARGED, shared across every build of this collection):")
    t_bisect = run("create-rgb", [CREATE_RGB, "--input", ciff, "--output-ciff", reordered], out)

    print("query and id conversion (not charged):")
    t_pisa = run("queries-pisa",
                 [sys.executable, CONV_QUERIES, args.queries_jsonl, pisa, "--scale", args.scale], out)
    t_ids = run("doc-ids", [sys.executable, EXTRACT_IDS,
                            "--jsonl", args.docs_jsonl, "--out", out / "doc_ids.npy"], out)
    t_ids += run("query-ids", [sys.executable, EXTRACT_IDS,
                               "--jsonl", args.queries_jsonl, "--out", out / "queries_ids.npy"], out)

    timings = {
        "measured_by": f"prepare_lsp_input.py on {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "out_dir": str(out),
        "query_scale": args.scale,
        "charged": {
            "ciff_quantization_s": quantization_s,
            "graph_bisection_s": t_bisect,
        },
        "not_charged": {
            "ciff_conversion_s": t_noquant,
            "queries_pisa_s": t_pisa,
            "extract_ids_s": t_ids,
        },
        "subtraction_evidence": {
            "ciff_noquant_a_s": t_noquant_a,
            "ciff_quantized_s": t_quant,
            "ciff_noquant_b_s": t_noquant_b,
            "bracket_spread_frac": bracket_spread,
            "valid": subtraction_valid,
        },
    }
    sidecar.write_text(json.dumps(timings, indent=2))

    print(f"\nwrote {sidecar}")
    print(f"  CHARGED: quantization {quantization_s:.1f}s + bisection {t_bisect:.1f}s "
          f"= {quantization_s + t_bisect:.1f}s per build")
    print(f"  not charged: conversion {t_noquant:.1f}s + pisa {t_pisa:.1f}s + ids {t_ids:.1f}s")
    if not subtraction_valid:
        print(f"\n  *** SUBTRACTION NOT VALID: unquantized brackets {t_noquant_a:.1f}s and "
              f"{t_noquant_b:.1f}s differ by {100*bracket_spread:.1f}%, quantization came out "
              f"{quantization_s:.1f}s. The difference is measuring machine state, not "
              f"quantization. Do NOT charge this number; separate the step physically instead.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
