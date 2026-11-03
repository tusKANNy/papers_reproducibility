#!/usr/bin/env python3
"""Convert a seismic-format sparse dataset to the .csr format SINDI
reads (vsag's sparse/scripts/sindi_index_{build,search}).

BigANN's reference CSR is not used, because it uses a slightly different encoding.
The CSR is written from the collection itself.

CSR on-disk layout (consumed by sindi_index_{build,search} via
read_sparse_vectors_from_csr_file in sparse/scripts/sindi_index_search.cpp:71):

    int64 sizes[3]              # (num_rows, num_cols, nnz)
    int64 indptr[num_rows + 1]
    int32 indices[nnz]
    float32 data[nnz]

`num_cols` is the vocabulary size = max_term_id + 1 (must be ≥ every id in
the corpus). We derive it from a full scan of base and queries combined, so
that one value bounds both files: `sindi_index_search` fails if a query
references an id ≥ num_cols.
"""
import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


def strip_leading_count(bin_path: Path) -> tuple[np.ndarray, int]:
    """mmap the file, peel off the leading u32 count, return (payload_bytes, declared_count)."""
    size = os.stat(bin_path).st_size
    if size < 4:
        raise ValueError(f"{bin_path}: file too short ({size} bytes)")
    mm = np.memmap(bin_path, dtype=np.uint8, mode="r", shape=(size,))
    declared = int(np.frombuffer(mm[:4].tobytes(), dtype="<u4")[0])
    return mm[4:], declared


def scan_payload(payload: np.ndarray) -> tuple[int, int, int]:
    """Walk the seismic byte stream, return (vector_count, max_term_id, nnz_total)."""
    raw = np.ascontiguousarray(payload).view(np.uint8)
    n = raw.nbytes
    i = 0
    count = 0
    max_id = 0
    nnz_total = 0
    while i < n:
        ln = int(raw[i : i + 4].view("<u4")[0])
        i += 4
        if ln > 0:
            id_end = i + 4 * ln
            val_end = id_end + 4 * ln
            if val_end > n:
                raise ValueError("truncated vector body")
            ids = raw[i:id_end].view("<u4")
            m = int(ids.max())
            if m > max_id:
                max_id = m
            nnz_total += ln
            i = val_end
        count += 1
    return count, max_id, nnz_total


def pack_csr(
    payload: np.ndarray, num_rows: int, nnz_total: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second pass: fill (indptr, indices, data) directly from the byte stream.

    Avoids per-doc Python loops with array copies: each doc's (ids, vals)
    slice is copied into the preallocated CSR arrays. The ids in seismic
    are u32 LE; CSR needs i32. Negative ids would corrupt this cast, but the
    vocabulary fits in 30k, far below INT32_MAX.
    """
    raw = np.ascontiguousarray(payload).view(np.uint8)
    n = raw.nbytes
    indptr = np.empty(num_rows + 1, dtype=np.int64)
    indices = np.empty(nnz_total, dtype=np.int32)
    data = np.empty(nnz_total, dtype=np.float32)
    indptr[0] = 0

    i = 0
    cursor = 0
    doc = 0
    while i < n:
        ln = int(raw[i : i + 4].view("<u4")[0])
        i += 4
        if ln > 0:
            id_end = i + 4 * ln
            val_end = id_end + 4 * ln
            indices[cursor : cursor + ln] = raw[i:id_end].view("<u4").astype(np.int32)
            data[cursor : cursor + ln] = raw[id_end:val_end].view("<f4")
            cursor += ln
            i = val_end
        doc += 1
        indptr[doc] = cursor
    if doc != num_rows:
        raise ValueError(f"pack_csr: walked {doc} docs but expected {num_rows}")
    if cursor != nnz_total:
        raise ValueError(f"pack_csr: filled {cursor} nnz but expected {nnz_total}")
    return indptr, indices, data


def write_csr(
    out_path: Path,
    num_rows: int,
    num_cols: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
) -> None:
    nnz = int(indices.size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack("<qqq", num_rows, num_cols, nnz))
        # numpy default byte order is native (== LE on x86), tofile writes raw
        # bytes. Force LE explicitly to stay portable.
        indptr.astype("<i8", copy=False).tofile(f)
        indices.astype("<i4", copy=False).tofile(f)
        data.astype("<f4", copy=False).tofile(f)


def convert_one(
    bin_path: Path, out_path: Path, num_cols: int
) -> tuple[int, int, int]:
    payload, declared = strip_leading_count(bin_path)
    count, _, nnz_total = scan_payload(payload)
    if count != declared:
        raise SystemExit(f"{bin_path}: declared={declared} but walked {count}")
    indptr, indices, data = pack_csr(payload, count, nnz_total)
    write_csr(out_path, count, num_cols, indptr, indices, data)
    return count, num_cols, nnz_total


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seismic-dir",
        required=True,
        help="Directory containing documents.bin and queries.bin (and optional groundtruth.tsv).",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory. Writes base.csr and queries.csr inside it.",
    )
    p.add_argument(
        "--num-cols-override",
        type=int,
        default=None,
        help="Force num_cols (= vocabulary size). Default: derived from max_term_id+1 over base ∪ queries.",
    )
    args = p.parse_args()

    seismic_dir = Path(args.seismic_dir)
    out_dir = Path(args.out_dir)
    docs_bin = seismic_dir / "documents.bin"
    qrys_bin = seismic_dir / "queries.bin"
    for f in (docs_bin, qrys_bin):
        if not f.exists():
            print(f"missing input: {f}", file=sys.stderr)
            return 1

    base_out = out_dir / "base.csr"
    qrys_out = out_dir / "queries.csr"

    # Two scans up-front to derive num_cols deterministically across both files.
    print(f"scanning {docs_bin} for vocab/nnz …")
    docs_payload, n_docs = strip_leading_count(docs_bin)
    n_docs_w, max_d_term, _ = scan_payload(docs_payload)
    if n_docs_w != n_docs:
        raise SystemExit(f"{docs_bin}: declared={n_docs} but walked {n_docs_w}")

    print(f"scanning {qrys_bin} for vocab/nnz …")
    qrys_payload, n_qrys = strip_leading_count(qrys_bin)
    n_qrys_w, max_q_term, _ = scan_payload(qrys_payload)
    if n_qrys_w != n_qrys:
        raise SystemExit(f"{qrys_bin}: declared={n_qrys} but walked {n_qrys_w}")

    max_term_id = max(max_d_term, max_q_term)
    num_cols = args.num_cols_override or (max_term_id + 1)
    print(
        f"  base: {n_docs:,} docs  queries: {n_qrys:,} docs  "
        f"max_term_id={max_term_id}  -> num_cols={num_cols}"
    )

    print(f"writing {base_out}")
    n, _, base_nnz = convert_one(docs_bin, base_out, num_cols)
    print(f"  rows={n:,}  nnz={base_nnz:,}  size={base_out.stat().st_size:,} B")

    print(f"writing {qrys_out}")
    n, _, qrys_nnz = convert_one(qrys_bin, qrys_out, num_cols)
    print(f"  rows={n:,}  nnz={qrys_nnz:,}  size={qrys_out.stat().st_size:,} B")

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
