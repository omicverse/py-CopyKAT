"""Generate the 4-panel pycopykat-vs-R-copykat comparison figure and metrics.

Panels:
    1. R copykat class (aneuploid/diploid/not.defined) on the sliced UMAP.
    2. Agreement between py and R (both_aneuploid / R_only / py_only /
       both_diploid / undefined).
    3. pycopykat class on the same UMAP.
    4. per-cell CNA intensity (mean |CNA|), py or R depending on --intensity.

The UMAP coordinates come from ``cells.csv`` — they are layout only and
carry no truth-label information. The 3CA ``cell_type`` column, if present,
is not used: this script only measures pycopykat ↔ R copykat consistency
under unsupervised auto-baseline.

Writes a single-row ``metrics.csv`` with columns:
    comparison=py_vs_r, ari, kappa, fmi, n_cells, py_min, r_min, speedup
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pycopykat.validation.metrics import compare_predictions


CLASS_COLORS = {
    "aneuploid":   "#C0392B",
    "diploid":     "#F2C94C",
    "not.defined": "#B0B0B0",
    "missing":     "#B0B0B0",
}

AGREE_COLORS = {
    "both_aneuploid": "#4C72B0",
    "R_only":         "#55A868",
    "py_only":        "#DD8452",
    "both_diploid":   "#F2C94C",
    "undefined":      "#B0B0B0",
}


def _normalise_pred_labels(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(
        r"^c[12]:(diploid|aneuploid):low\.conf$", r"\1", regex=True
    )


def _read_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] != 2:
        raise ValueError(f"expected 2 cols in {path}")
    df.columns = ["cell", "pred"]
    df["pred"] = _normalise_pred_labels(df["pred"])
    return df


def _cna_intensity(cna_path: Path) -> pd.Series:
    df = pd.read_csv(cna_path, sep="\t")
    cell_cols = df.columns.tolist()
    # first N columns are bin coords (chrom, chrompos/start, [end]); drop them
    leading = 3 if "end" in cell_cols[:5] else 3
    cell_cols = df.columns[leading:].tolist()
    return df[cell_cols].abs().mean(axis=0)


def _build_agreement(merged: pd.DataFrame) -> pd.Series:
    agree = pd.Series("undefined", index=merged.index, dtype=object)
    r_pos = merged["r_class"] == "aneuploid"
    r_neg = merged["r_class"] == "diploid"
    p_pos = merged["py_class"] == "aneuploid"
    p_neg = merged["py_class"] == "diploid"
    agree[r_pos & p_pos] = "both_aneuploid"
    agree[r_pos & p_neg] = "R_only"
    agree[r_neg & p_pos] = "py_only"
    agree[r_neg & p_neg] = "both_diploid"
    return agree


def _scatter_categorical(
    ax, x, y, labels, palette, title, point_size, legend_loc="center left",
):
    order = list(palette.keys())
    background = {"not.defined", "undefined", "both_diploid", "missing"}
    back = [c for c in order if c in background]
    fore = [c for c in order if c not in background]
    for cat in back + fore:
        if cat not in labels.values:
            continue
        m = (labels == cat).to_numpy()
        ax.scatter(
            x[m], y[m], color=palette[cat], s=point_size, linewidths=0,
            label=f"{cat} (n={int(m.sum())})",
        )
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title, fontsize=11)
    ax.legend(
        loc=legend_loc, bbox_to_anchor=(1.02, 0.5),
        frameon=False, fontsize=8, handletextpad=0.3,
        borderaxespad=0.0,
    )


def _scatter_continuous(ax, x, y, values, title, cmap, point_size, cbar_label):
    order = np.argsort(values)
    sc = ax.scatter(
        x[order], y[order], c=values[order], cmap=cmap,
        s=point_size, linewidths=0,
    )
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title, fontsize=11)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _read_runinfo_minutes(runinfo_path: Path) -> float:
    if not runinfo_path.exists():
        return float("nan")
    for line in runinfo_path.read_text().splitlines():
        if line.startswith("elapsed_min="):
            try:
                return float(line.split("=", 1)[1])
            except ValueError:
                return float("nan")
    return float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--r-out", type=Path, required=True)
    p.add_argument("--py-out", type=Path, required=True)
    p.add_argument("--cells", type=Path, required=True,
                   help="cells.csv with cell_name, umap1, umap2 (layout only)")
    p.add_argument("--sam-name", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--title-suffix", default="")
    p.add_argument("--point-size", type=float, default=8.0)
    p.add_argument("--intensity", choices=["py", "r"], default="py")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    r_pred = _read_prediction(args.r_out / f"{args.sam_name}_copykat_prediction.txt")
    py_pred = _read_prediction(
        args.py_out / f"{args.sam_name}_py_copykat_prediction.txt"
    )
    r_pred = r_pred.rename(columns={"pred": "r_class"})
    py_pred = py_pred.rename(columns={"pred": "py_class"})

    cells = pd.read_csv(args.cells)

    merged = cells.merge(r_pred, left_on="cell_name", right_on="cell", how="left")
    merged = merged.drop(columns=["cell"]).merge(
        py_pred, left_on="cell_name", right_on="cell", how="left"
    ).drop(columns=["cell"])
    merged["r_class"] = merged["r_class"].fillna("not.defined")
    merged["py_class"] = merged["py_class"].fillna("not.defined")
    merged["agreement"] = _build_agreement(merged)

    # CNA intensity (py by default; falls back to R if py file missing)
    try:
        if args.intensity == "py":
            intens = _cna_intensity(
                args.py_out / f"{args.sam_name}_py_copykat_CNA_results.txt"
            )
        else:
            intens = _cna_intensity(
                args.r_out / f"{args.sam_name}_copykat_CNA_results.txt"
            )
        merged["intensity"] = merged["cell_name"].map(intens).astype(float)
    except Exception as e:
        print(f"[warn] could not compute intensity: {e}")
        merged["intensity"] = 0.0
    merged["intensity"] = merged["intensity"].fillna(merged["intensity"].median())

    # --- metrics (pycopykat vs R copykat only) ---
    py_vs_r = compare_predictions(
        r_pred.rename(columns={"r_class": "copykat.pred"}),
        py_pred.rename(columns={"py_class": "copykat.pred"}),
    )

    py_min = _read_runinfo_minutes(
        args.py_out / f"{args.sam_name}_py_copykat_runinfo.txt"
    )
    r_min = _read_runinfo_minutes(
        args.r_out / f"{args.sam_name}_copykat_runinfo.txt"
    )
    speedup = float("nan")
    if np.isfinite(py_min) and np.isfinite(r_min) and py_min > 0:
        speedup = r_min / py_min

    metrics_row = {
        "comparison": "py_vs_r",
        "ari":        py_vs_r["ari"],
        "kappa":      py_vs_r["kappa"],
        "fmi":        py_vs_r["fmi"],
        "n_cells":    py_vs_r["n_shared"],
        "py_min":     py_min,
        "r_min":      r_min,
        "speedup":    speedup,
    }
    pd.DataFrame([metrics_row]).to_csv(args.out_dir / "metrics.csv", index=False)

    # --- figure ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    x = merged["umap1"].to_numpy()
    y = merged["umap2"].to_numpy()

    _scatter_categorical(
        axes[0, 0], x, y, merged["r_class"], CLASS_COLORS,
        title="copykat_R_class", point_size=args.point_size,
    )
    _scatter_categorical(
        axes[0, 1], x, y, merged["agreement"], AGREE_COLORS,
        title="agreement_R_vs_py", point_size=args.point_size,
    )
    _scatter_categorical(
        axes[1, 0], x, y, merged["py_class"], CLASS_COLORS,
        title="pycopykat_class", point_size=args.point_size,
    )
    label = "py" if args.intensity == "py" else "R"
    _scatter_continuous(
        axes[1, 1], x, y, merged["intensity"].to_numpy(),
        title=f"{label}_CNA_intensity (mean |CNA|)",
        cmap="magma", point_size=args.point_size,
        cbar_label="|CNA|",
    )

    suffix = f"  —  {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(
        f"pycopykat vs R copykat (ARI={py_vs_r['ari']:.3f}, "
        f"κ={py_vs_r['kappa']:.3f}, FMI={py_vs_r['fmi']:.3f})"
        f"{suffix}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig_path = args.out_dir / "figure_py_vs_R.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"[compare] wrote {fig_path}")
    print(f"[compare] wrote {args.out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
