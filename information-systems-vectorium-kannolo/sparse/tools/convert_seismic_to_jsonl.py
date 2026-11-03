#!/usr/bin/env python3
"""Convert a seismic-format `.bin` sparse file into the JSONL that LSP's input preparation reads.

    seismic .bin  ->  {"id": "<positional index>", "vector": {"<component>": <score>, ...}}

The seismic collections ship `documents.bin` / `queries.bin`; DSP's pipeline starts from JSONL.
This script converts one into the other.

**Ids are the positional index, written as a string. Correctness depends on this choice.**
The seismic format carries no ids: a document is identified by its position in the file, and the
shipped `groundtruth.tsv` uses the same positional ids. DSP carries the `id` written here
all the way through: `jsonl2ciff` stores it in the CIFF document lexicon, `create-rgb` permutes
document *positions* while the lexicon travels with them, and `to_trec` prints
`doc_lexicon[internal_id]` (`src/util.rs:25`). So the TREC output comes back in the same space the
ground truth uses, and no `doc_ids.npy` round-trip is needed to read it.

If `create-rgb` reordered the corpus without the lexicon, the run would look plausible but
every id would be wrong, and no internal consistency check would show it. Verify a known query's top-10
against the shipped ground truth before trusting any DSP accuracy number.

**Terms are the component ids, serialized as decimal strings.** The collections ship no
`token_to_id_mapping.json`, and CIFF terms are opaque strings, so each integer component id is
used as its own term. Documents and queries must be converted by this same script so that the
two lexicons agree. "10" and "9" sort lexicographically inside CIFF, which is harmless because
every lookup is by string.

**Precision.** Values are written with 7 significant digits. f32 carries ~7.2 decimal digits, so
this is lossless for the stored data and far smaller than Python's 17-digit repr. It is also much
finer than `jsonl2ciff --quantize`'s 8-bit impact quantization, which LSP requires.

Size: NQ's 411.9 M document nnz take roughly 9 GB of JSON.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def iter_seismic(path, limit=None):
    """Yield `(index, components, values)` per vector.

    The file is read once into a byte buffer and sliced. Three reads per vector would be slow
    for 2.68 M vectors. NQ's documents.bin is 3.3 GB,
    small next to the 503 GiB of the benchmark machine, but the buffer is released per caller
    and not held alongside a growing output.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 4:
        raise SystemExit(f"{path}: too short to hold a header")
    n = int(raw[:4].view(np.uint32)[0])
    if limit is not None:
        n = min(n, limit)
    pos = 4
    for i in range(n):
        nnz = int(raw[pos:pos + 4].view(np.uint32)[0])
        pos += 4
        comps = raw[pos:pos + 4 * nnz].view(np.uint32)
        vals = raw[pos + 4 * nnz:pos + 8 * nnz].view(np.float32)
        pos += 8 * nnz
        yield i, comps, vals


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seismic", required=True, type=Path, help="documents.bin or queries.bin")
    p.add_argument("--out", required=True, type=Path, help="output .jsonl")
    p.add_argument("--limit", type=int, default=None, help="first N vectors only (smoke tests)")
    p.add_argument("--batch", type=int, default=10000, help="lines buffered per write")
    args = p.parse_args()

    if not args.seismic.is_file():
        print(f"ERROR: not found: {args.seismic}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_vec = n_nnz = n_nonpos = 0
    max_component = -1
    buf = []
    with args.out.open("w") as fout:
        for i, comps, vals in iter_seismic(args.seismic, args.limit):
            # A non-positive impact has no meaning in an impact-ordered inverted index, and
            # `jsonl2ciff --quantize` cannot represent one. Such values are dropped, but counted
            # and reported: if a collection ever has any, that is a finding about the collection,
            # and the count makes it visible.
            keep = vals > 0.0
            n_nonpos += int(comps.size - keep.sum())
            c, v = comps[keep], vals[keep]
            if c.size:
                max_component = max(max_component, int(c.max()))
            vector = {str(int(t)): float(f"{float(s):.7g}") for t, s in zip(c, v)}
            buf.append(json.dumps({"id": str(i), "vector": vector},
                                  separators=(",", ":"), ensure_ascii=False))
            n_vec += 1
            n_nnz += int(c.size)
            if len(buf) >= args.batch:
                fout.write("\n".join(buf) + "\n")
                buf.clear()
                if n_vec % 500000 == 0:
                    print(f"  {n_vec:,} vectors", file=sys.stderr, flush=True)
        if buf:
            fout.write("\n".join(buf) + "\n")

    print(f"wrote {n_vec:,} vectors, {n_nnz:,} nnz, max component {max_component} -> {args.out}")
    print(f"  size {args.out.stat().st_size / 2**30:.2f} GiB")
    if n_nonpos:
        print(f"  WARNING: dropped {n_nonpos:,} non-positive values; investigate before "
              "trusting any accuracy number from this file", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
