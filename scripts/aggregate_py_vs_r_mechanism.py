"""Derive a truth-free mechanism classification from the py↔R summary.

Reads ``benchmarks/full/py_vs_r_summary.csv`` (produced by run_py_sweep.py)
and writes ``mechanism_summary.csv`` with per-patient ARI, κ, FMI, runtime,
and a coarse mechanism label that uses ONLY py↔R agreement — no 3CA truth.

Mechanism classification:
    * ari >= 0.9 and kappa >= 0.9  -> "matched"
      (py and R produce near-identical partitions with consistent labels)
    * ari >= 0.5 and kappa < 0     -> "R_label_flipped"
      (same partition but diploid/aneuploid labels inversely assigned)
    * ari < 0.5                    -> "different_clusters"
      (structural disagreement on the 2-way partition)
    * otherwise                    -> "partial"
      (moderate ARI without a clean label flip)

Output columns:
    cancer, sample, n_cells, py_vs_r_ari, py_vs_r_kappa, py_vs_r_fmi,
    mechanism, py_min, r_min, speedup
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _classify(ari: float, kappa: float) -> str:
    if pd.isna(ari) or pd.isna(kappa):
        return "unknown"
    if ari >= 0.9 and kappa >= 0.9:
        return "matched"
    if ari >= 0.5 and kappa < 0:
        return "R_label_flipped"
    if ari < 0.5:
        return "different_clusters"
    return "partial"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True,
                    help="benchmarks/full/py_vs_r_summary.csv")
    ap.add_argument("--out", type=Path, required=True,
                    help="output mechanism_summary.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    rows = []
    for _, r in df.iterrows():
        ari = r["py_vs_r__ari"]
        kappa = r["py_vs_r__kappa"]
        fmi = r.get("py_vs_r__fmi", float("nan"))
        n = int(r["py_vs_r__n_cells"]) if pd.notna(r["py_vs_r__n_cells"]) else 0
        py_min = r.get("py_vs_r__py_min", float("nan"))
        r_min = r.get("py_vs_r__r_min", float("nan"))
        speedup = r.get("py_vs_r__speedup", float("nan"))
        rows.append({
            "cancer":        r["cancer"],
            "sample":        r["sample"],
            "n_cells":       n,
            "py_vs_r_ari":   ari,
            "py_vs_r_kappa": kappa,
            "py_vs_r_fmi":   fmi,
            "mechanism":     _classify(ari, kappa),
            "py_min":        py_min,
            "r_min":         r_min,
            "speedup":       speedup,
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"[agg] wrote {args.out}  ({len(out)} patients)")
    print(out["mechanism"].value_counts().to_string())


if __name__ == "__main__":
    main()
