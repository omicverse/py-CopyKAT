"""Slice multiple patients out of one 3CA dataset in a single mtx read.

Memory-conscious variant of ``slice_patient.py``: loads the sparse
``Exp_data_UMIcounts.mtx`` exactly once, writes one ``<patient>/counts.tsv``
per requested patient (plus a matching ``cells.csv``), and frees the full
matrix when done.

Usage::

    uv run python scripts/slice_dataset.py \\
        --dataset tests/data/Data_Kim2020_Lung \\
        --samples P1028,P0019,P0034 \\
        --out-base benchmarks/full/Kim2020_Lung
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread


def slice_dataset(
    dataset: Path,
    samples: list[str],
    out_base: Path,
    counts_file: str = "Exp_data_UMIcounts.mtx",
    genes_file: str = "Genes.txt",
    cells_file: str = "Cells.csv",
) -> None:
    cells = pd.read_csv(dataset / cells_file)
    # 3CA's Genes.txt for some datasets (Gao2021_Breast, Obradovic2021_Kidney)
    # ships each symbol wrapped in double quotes, e.g. "RP11-34P13.3".
    # Strip a matching leading/trailing quote pair so R read.table doesn't
    # see triple-quoted row names after pandas re-quotes them.
    def _clean(g: str) -> str:
        g = g.strip()
        if len(g) >= 2 and g[0] == g[-1] and g[0] in ('"', "'"):
            g = g[1:-1]
        return g

    genes = [
        _clean(g)
        for g in (dataset / genes_file).read_text().splitlines()
        if g.strip()
    ]
    if "sample" not in cells.columns:
        raise ValueError(f"'sample' missing from {cells_file}")

    print(f"[slice] reading {dataset / counts_file} (single pass)…")
    M = mmread(str(dataset / counts_file)).tocsc()
    if M.shape[0] != len(genes):
        raise ValueError(f"mtx rows {M.shape[0]} != genes {len(genes)}")
    if M.shape[1] != len(cells):
        raise ValueError(f"mtx cols {M.shape[1]} != cells {len(cells)}")

    # Coerce sample column to string so numeric-looking IDs (e.g. Qian "11")
    # compare correctly against CLI-supplied string arguments.
    sample_str = cells["sample"].astype(str)
    for sample in samples:
        mask = (sample_str == sample).to_numpy()
        n = int(mask.sum())
        if n == 0:
            print(f"[slice] !! sample {sample!r} not found, skipping")
            continue
        out_dir = out_base / sample
        out_dir.mkdir(parents=True, exist_ok=True)

        M_sub = M[:, np.where(mask)[0]]
        cell_names = cells.loc[mask, "cell_name"].tolist()
        print(f"[slice] {sample}: {M_sub.shape[0]} genes × {n} cells")

        dense = pd.DataFrame(
            M_sub.toarray().astype(np.int32),
            index=genes,
            columns=cell_names,
        )
        dense.index.name = "gene"
        counts_path = out_dir / "counts.tsv"
        dense.to_csv(counts_path, sep="\t")
        size_mb = counts_path.stat().st_size / 1e6
        print(f"[slice]   wrote {counts_path} ({size_mb:.1f} MB)")
        del dense, M_sub

        meta_path = out_dir / "cells.csv"
        cells.loc[mask].to_csv(meta_path, index=False)
        print(f"[slice]   wrote {meta_path}")

    del M
    print("[slice] done.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--samples", required=True,
                   help="comma-separated patient ids, e.g. P1028,P0019,P0034")
    p.add_argument("--out-base", type=Path, required=True)
    args = p.parse_args()
    samples = [s.strip() for s in args.samples.split(",") if s.strip()]
    slice_dataset(args.dataset, samples, args.out_base)


if __name__ == "__main__":
    main()
