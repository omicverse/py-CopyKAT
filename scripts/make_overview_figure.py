"""Cancer-level overview of the pycopykat vs R copykat benchmark.

Reads ``benchmarks/full/py_vs_r_summary.csv`` (per-patient py↔R metrics
produced by run_py_sweep.py) and produces four panels in a single PNG:

* Panel A — per-patient py↔R agreement (ARI, κ, FMI).
* Panel B — per-patient runtime (R vs pycopykat minutes) with speedup
  factor annotated.
* Panel C — distribution of py↔R ARI across all patients.
* Panel D — per-cancer mean py↔R ARI and mean speedup.

No truth labels are used; this script visualises pycopykat ↔ R copykat
consistency only.

Usage::

    uv run python scripts/make_overview_figure.py \\
        --summary benchmarks/full/py_vs_r_summary.csv \\
        --out benchmarks/full/overview.png
"""
from __future__ import annotations

import argparse
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


def _barplot_grouped(ax, df, x_label, palette, ylabel, title, y_max=None):
    n = len(df)
    w = 0.9 / max(len(palette), 1)
    x = np.arange(n)
    for i, (col, colour) in enumerate(palette.items()):
        offset = (i - (len(palette) - 1) / 2) * w
        ax.bar(x + offset, df[col].to_numpy(), width=w, color=colour, label=col)
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

    # Clip kappa to [0, 1] for plotting alongside ARI; negative kappa is a
    # label-flip mechanism flag, which Panel C highlights separately.
    summary["py_vs_r__kappa_clipped"] = summary["py_vs_r__kappa"].clip(lower=0.0)

    fig, axes = plt.subplots(
        2, 2, figsize=(max(11, 0.55 * len(summary) + 6), 10),
    )
    axA, axB, axC, axD = axes.flat

    # Panel A: py↔R agreement metrics
    _barplot_grouped(
        axA, summary, x_label="",
        palette={
            "py_vs_r__ari":           "#4C72B0",
            "py_vs_r__kappa_clipped": "#55A868",
            "py_vs_r__fmi":           "#E67E22",
        },
        ylabel="metric",
        title="A. pycopykat ↔ R copykat agreement (ARI / κ₊ / FMI)",
        y_max=1.05,
    )

    # Panel B: runtime (bars for R and py; annotated speedup)
    r_col = "py_vs_r__r_min" if "py_vs_r__r_min" in summary.columns else None
    py_col = "py_vs_r__py_min" if "py_vs_r__py_min" in summary.columns else None
    sp_col = "py_vs_r__speedup" if "py_vs_r__speedup" in summary.columns else None
    if r_col and py_col and summary[r_col].notna().any() and summary[py_col].notna().any():
        _barplot_grouped(
            axB, summary, x_label="patient",
            palette={
                r_col:  "#C0392B",
                py_col: "#4C72B0",
            },
            ylabel="runtime (minutes)",
            title="B. Runtime (R vs pycopykat)",
        )
        if sp_col:
            for i, s in enumerate(summary[sp_col]):
                if pd.notna(s):
                    top = max(
                        summary.loc[i, r_col] or 0.0,
                        summary.loc[i, py_col] or 0.0,
                    )
                    axB.text(
                        i, top + 0.3,
                        f"{s:.1f}×", ha="center", fontsize=8, color="#333",
                    )
    else:
        axB.text(0.5, 0.5, "no runtime data", transform=axB.transAxes, ha="center")

    # Panel C: ARI distribution
    aris = summary["py_vs_r__ari"].dropna().to_numpy()
    axC.hist(aris, bins=np.linspace(0, 1, 21), color="#4C72B0", edgecolor="white")
    axC.axvline(np.median(aris), color="#C0392B", linestyle="--", linewidth=1,
                label=f"median={np.median(aris):.3f}")
    axC.axvline(np.mean(aris), color="#27AE60", linestyle=":", linewidth=1,
                label=f"mean={np.mean(aris):.3f}")
    axC.set_xlabel("py↔R ARI")
    axC.set_ylabel("patients")
    axC.set_title("C. py↔R ARI distribution (17 patients)", fontsize=11)
    axC.legend(fontsize=8, frameon=False)
    axC.grid(axis="y", alpha=0.25)

    # Panel D: per-cancer mean ARI and speedup
    agg_cols = {
        "n_patients":         ("sample", "size"),
        "py_vs_r_ari_mean":   ("py_vs_r__ari", "mean"),
        "py_vs_r_kappa_mean": ("py_vs_r__kappa", "mean"),
        "py_vs_r_fmi_mean":   ("py_vs_r__fmi", "mean"),
    }
    if r_col and py_col:
        agg_cols["r_min_mean"] = (r_col, "mean")
        agg_cols["py_min_mean"] = (py_col, "mean")
    if sp_col:
        agg_cols["speedup_mean"] = (sp_col, "mean")
        agg_cols["speedup_median"] = (sp_col, "median")
    agg = summary.groupby("cancer").agg(**agg_cols)
    agg = agg.loc[sorted(agg.index)]

    x = np.arange(len(agg))
    w = 0.4
    ax2 = axD.twinx()
    bars_ari = axD.bar(
        x - w / 2, agg["py_vs_r_ari_mean"].to_numpy(),
        width=w, color="#4C72B0", label="mean ARI",
    )
    if "speedup_mean" in agg.columns:
        bars_sp = ax2.bar(
            x + w / 2, agg["speedup_mean"].to_numpy(),
            width=w, color="#E67E22", label="mean speedup (×)",
        )
    axD.set_xticks(x)
    axD.set_xticklabels(agg.index, rotation=30, ha="right", fontsize=9)
    for tick, c in zip(axD.get_xticklabels(), agg.index):
        tick.set_color(CANCER_COLORS.get(c, "#333333"))
    axD.set_ylabel("mean py↔R ARI", color="#4C72B0")
    axD.set_ylim(0, 1.05)
    ax2.set_ylabel("mean speedup (R min / py min)", color="#E67E22")
    axD.set_title("D. Per-cancer mean ARI and speedup", fontsize=11)
    axD.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"pycopykat vs R copykat benchmark — {len(summary)} patients across "
        f"{summary['cancer'].nunique()} cancers",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[overview] wrote {args.out}")

    agg_path = args.out.with_suffix(".cancer_summary.csv")
    agg.to_csv(agg_path)
    print(f"[overview] wrote {agg_path}")


if __name__ == "__main__":
    main()
