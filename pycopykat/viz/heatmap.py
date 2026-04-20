"""matplotlib CNA heatmap with a chromosome side bar.

Rows = cells grouped by ``copykat.pred`` then sorted alphabetically. Columns =
genomic bins in input order. A thin coloured strip above the heatmap shows
chromosome assignments; the heatmap uses ``RdBu_r`` centred at 0.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from matplotlib.colors import TwoSlopeNorm  # noqa: E402


def plot_cna_heatmap(
    cna: pd.DataFrame,
    prediction: pd.DataFrame,
    output: Path,
    *,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    """Save a CNA heatmap PNG to ``output``.

    Parameters
    ----------
    cna
        ``(n_bins, n_cells)`` DataFrame. If its index is a ``MultiIndex`` whose
        first level is chromosome, a chromosome-colour strip is drawn.
    prediction
        DataFrame with columns ``cell`` and ``copykat.pred``; used to sort rows.
    output
        Output PNG path.
    """
    pred_map = dict(zip(prediction["cell"], prediction["copykat.pred"]))
    col_order = sorted(cna.columns, key=lambda c: (pred_map.get(c, "zzz"), str(c)))
    M = cna[col_order].to_numpy().T  # cells × bins

    if isinstance(cna.index, pd.MultiIndex):
        chrom = cna.index.get_level_values(0).to_numpy()
    else:
        chrom = np.zeros(M.shape[1])

    fig, (ax_cb, ax_heat) = plt.subplots(
        2, 1, figsize=(12, 10),
        gridspec_kw={"height_ratios": [0.5, 20], "hspace": 0.02},
    )

    uniq = np.unique(chrom)
    cmap_chr = plt.get_cmap("tab20")(np.linspace(0, 1, len(uniq)))
    chr_colors = np.zeros((1, M.shape[1], 4))
    for k, u in enumerate(uniq):
        chr_colors[0, chrom == u] = cmap_chr[k]
    ax_cb.imshow(chr_colors, aspect="auto", interpolation="nearest")
    ax_cb.set_yticks([])
    ax_cb.set_xticks([])
    ax_cb.set_title("chromosome", fontsize=9, pad=4)

    ax_heat.imshow(
        M, aspect="auto", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
    )
    ax_heat.set_ylabel(f"{M.shape[0]} cells (sorted by pred)")
    ax_heat.set_xlabel(f"{M.shape[1]} bins")
    ax_heat.set_yticks([])
    ax_heat.set_xticks([])

    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
