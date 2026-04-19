"""Run pycopykat on a sliced counts.tsv and write R-compatible outputs.

Produces ``<out_dir>/<sam_name>_py_copykat_prediction.txt`` and
``<sam_name>_py_copykat_CNA_results.txt`` alongside a ``run.log``.

Usage::

    uv run python scripts/run_py_copykat.py \\
        --counts benchmarks/pilot/Kim2020_Lung_P0028/counts.tsv \\
        --out benchmarks/pilot/Kim2020_Lung_P0028/py_out \\
        --sam-name P0028_pilot --n-jobs 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from pycopykat import CopykatConfig, copykat


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--counts", type=Path, required=True, help="genes x cells TSV")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--sam-name", required=True)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--min-gene-per-cell", type=int, default=200)
    p.add_argument("--low-dr", type=float, default=0.05)
    p.add_argument("--up-dr", type=float, default=0.1)
    p.add_argument("--win-size", type=int, default=25)
    p.add_argument("--ks-cut", type=float, default=0.1)
    p.add_argument("--distance", default="euclidean")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[py] reading counts: {args.counts}")
    mat = pd.read_csv(args.counts, sep="\t", index_col=0)
    print(f"[py] dim: {mat.shape[0]} genes × {mat.shape[1]} cells")

    cfg = CopykatConfig(
        min_gene_per_cell=args.min_gene_per_cell,
        low_dr=args.low_dr,
        up_dr=args.up_dr,
        win_size=args.win_size,
        ks_cut=args.ks_cut,
        distance=args.distance,
        n_jobs=args.n_jobs,
        sam_name=args.sam_name,
        seed=args.seed,
        output_dir=args.out,
    )

    t0 = time.time()
    result = copykat(mat, config=cfg)
    elapsed = (time.time() - t0) / 60.0
    print(f"[py] copykat elapsed: {elapsed:.2f} min")
    print(f"[py] warnings: {result.warnings}")

    # Write R-compatible output files (prefixed "py_copykat_" to distinguish
    # from real R outputs sharing the same directory in some workflows)
    pred_path = args.out / f"{args.sam_name}_py_copykat_prediction.txt"
    result.prediction.to_csv(pred_path, sep="\t", index=False)
    cna_path = args.out / f"{args.sam_name}_py_copykat_CNA_results.txt"
    result.cna_mat.reset_index().to_csv(cna_path, sep="\t", index=False)

    (args.out / f"{args.sam_name}_py_copykat_runinfo.txt").write_text(
        f"elapsed_min={elapsed:.4f}\nn_jobs={args.n_jobs}\n"
        f"warnings={'; '.join(result.warnings) if result.warnings else 'none'}\n"
    )
    print(f"[py] wrote {pred_path}")
    print(f"[py] wrote {cna_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
