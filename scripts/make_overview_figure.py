"""Cancer-level overview of the pycopykat vs R copykat benchmark.

Reads ``benchmarks/full/py_vs_r_summary.csv`` (per-patient metrics) and
produces three panels in a single PNG:

* Panel A — per-patient accuracy (py vs R vs truth) grouped by cancer.
* Panel B — per-patient py↔R agreement (ARI, κ, FMI).
* Panel C — per-patient runtime ratio (R minutes / py minutes) if
  ``*_runinfo.txt`` files are present.

Usage::

    uv run python scripts/make_overview_figure.py \\
        --summary benchmarks/full/py_vs_r_summary.csv \\
        --out benchmarks/full/overview.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CANCER_COLORS = {
    "Gao2021_Breast":       "#C0392B",
    "Kim2020_Lung":         "#2980B9",
    "Lee2020_Colorectal":   "#27AE60",
    "Obradovic2021_Kidney": "#8E44AD",
    "Qian2020_Ovarian":     "#E67E22",
}


def _read_runinfo(patient_dir: Path, sam_name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, fname in (
        ("r_min", f"r_out/{sam_name}_copykat_runinfo.txt"),
        ("py_min", f"py_out/{sam_name}_py_copykat_runinfo.txt"),
    ):
        p = patient_dir / fname
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if line.startswith("elapsed_min="):
                try:
                    out[label] = float(line.split("=", 1)[1])
                except ValueError:
                    pass
    return out


def build_runtime_frame(summary: pd.DataFrame, base: Path) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        pt_dir = base / r["cancer"] / r["sample"]
        info = _read_runinfo(pt_dir, f"{r['sample']}_full")
        rows.append({
            "cancer": r["cancer"],
            "sample": r["sample"],
            "r_min":  info.get("r_min", float("nan")),
            "py_min": info.get("py_min", float("nan")),
        })
    return pd.DataFrame(rows)


def _barplot_grouped(ax, df, x_label, value_cols, palette, ylabel, title, y_max=None):
    n = len(df)
    w = 0.32
    x = np.arange(n)
    for i, (col, colour) in enumerate(palette.items()):
        ax.bar(x + (i - 0.5) * w, df[col].to_numpy(), width=w, color=colour, label=col)
    # cancer-colored ticks
    labels = [f"{r['cancer'][:3]}/{r['sample']}" for _, r in df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    for tick, (_, r) in zip(ax.get_xticklabels(), df.iterrows()):
        tick.set_color(CANCER_COLORS.get(r["cancer"], "#333333"))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_label)
    ax.set_title(title, fontsize=11)
    if y_max is not None:
        ax.set_ylim(0, y_max)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    summary = pd.read_csv(args.summary)
    summary = summary.sort_values(["cancer", "sample"]).reset_index(drop=True)

    base = args.summary.parent
    rt = build_runtime_frame(summary, base)
    summary = summary.merge(rt, on=["cancer", "sample"], how="left")

    fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(max(10, 0.9 * len(summary) + 2), 11))

    # Panel A: accuracy vs truth
    _barplot_grouped(
        axA, summary, x_label="",
        value_cols=("py_vs_truth__accuracy", "r_vs_truth__accuracy"),
        palette={
            "py_vs_truth__accuracy": "#4C72B0",
            "r_vs_truth__accuracy":  "#C0392B",
        },
        ylabel="accuracy vs 3CA cell_type",
        title="A. Malignant/non-Malignant accuracy (py vs R, truth = 3CA labels)",
        y_max=1.05,
    )

    # Panel B: py-vs-R agreement metrics
    _barplot_grouped(
        axB, summary, x_label="",
        value_cols=("py_vs_r__ari", "py_vs_r__kappa", "py_vs_r__fmi"),
        palette={
            "py_vs_r__ari":   "#4C72B0",
            "py_vs_r__kappa": "#55A868",
            "py_vs_r__fmi":   "#E67E22",
        },
        ylabel="metric",
        title="B. pycopykat ↔ R copykat agreement (ARI / κ / FMI)",
        y_max=1.05,
    )

    # Panel C: runtime (bars for R and py; line for speedup)
    if rt["r_min"].notna().any() and rt["py_min"].notna().any():
        _barplot_grouped(
            axC, summary, x_label="patient",
            value_cols=("r_min", "py_min"),
            palette={
                "r_min":  "#C0392B",
                "py_min": "#4C72B0",
            },
            ylabel="runtime (minutes)",
            title="C. Runtime (R vs pycopykat)",
        )
        # annotate speedup factor
        speedup = summary["r_min"] / summary["py_min"].replace({0: np.nan})
        for i, s in enumerate(speedup):
            if pd.notna(s):
                axC.text(
                    i, max(summary.loc[i, "r_min"], summary.loc[i, "py_min"]) + 0.3,
                    f"{s:.1f}×", ha="center", fontsize=8, color="#333",
                )
    else:
        axC.text(0.5, 0.5, "no runtime data", transform=axC.transAxes, ha="center")

    fig.suptitle(
        f"pycopykat vs R copykat benchmark — {len(summary)} patients across "
        f"{summary['cancer'].nunique()} cancers",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[overview] wrote {args.out}")

    # Also produce a per-cancer aggregate table
    agg = summary.groupby("cancer").agg(
        n_patients=("sample", "size"),
        py_acc_mean=("py_vs_truth__accuracy", "mean"),
        r_acc_mean=("r_vs_truth__accuracy", "mean"),
        py_vs_r_ari_mean=("py_vs_r__ari", "mean"),
        py_vs_r_kappa_mean=("py_vs_r__kappa", "mean"),
        r_min_mean=("r_min", "mean"),
        py_min_mean=("py_min", "mean"),
    )
    agg["speedup"] = agg["r_min_mean"] / agg["py_min_mean"]
    agg_path = args.out.with_suffix(".cancer_summary.csv")
    agg.to_csv(agg_path)
    print(f"[overview] wrote {agg_path}")


if __name__ == "__main__":
    main()
