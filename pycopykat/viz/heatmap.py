"""matplotlib CNA heatmaps with chromosome side bars.

Three entry points:

* :func:`plot_cna_heatmap` — single-method heatmap (existing behaviour, callers
  may pass ``output=`` to save to PNG or ``ax=`` to draw inline).
* :func:`plot_cna_heatmap_compare` — py vs R side-by-side on the same cell
  ordering, for parity notebooks.
* :func:`plot_cna_delta` — pycopykat minus R bin-level difference heatmap.

Rows = cells grouped by ``copykat.pred`` then sorted alphabetically. Columns =
genomic bins in input order. The heatmap uses ``RdBu_r`` centred at 0.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm


def _chrom_strip(cna_index: pd.Index) -> np.ndarray:
    """Return a length-``n_bins`` chromosome-id array (zeros if no MultiIndex)."""
    if isinstance(cna_index, pd.MultiIndex):
        return cna_index.get_level_values(0).to_numpy()
    return np.zeros(len(cna_index))


def _draw_chrom_bar(ax: Axes, chrom: np.ndarray) -> None:
    uniq = np.unique(chrom)
    cmap_chr = plt.get_cmap("tab20")(np.linspace(0, 1, len(uniq)))
    chr_colors = np.zeros((1, len(chrom), 4))
    for k, u in enumerate(uniq):
        chr_colors[0, chrom == u] = cmap_chr[k]
    ax.imshow(chr_colors, aspect="auto", interpolation="nearest")
    ax.set_yticks([])
    ax.set_xticks([])


def _order_cells_by_pred(prediction: pd.DataFrame, cell_cols: list[str]) -> list[str]:
    """Sort ``cell_cols`` by ``copykat.pred`` (then cell name) using ``prediction``.

    Accepts predictions with either ``cell`` or ``cell.names`` column (R style)
    and either clean or ``low.conf``-suffixed labels. Cells missing from the
    prediction table are sorted last under sentinel ``"zzz"``.
    """
    cell_col = "cell" if "cell" in prediction.columns else "cell.names"
    pred_map = dict(zip(prediction[cell_col].astype(str), prediction["copykat.pred"].astype(str)))
    return sorted(cell_cols, key=lambda c: (pred_map.get(c, "zzz"), str(c)))


def plot_cna_heatmap(
    cna: pd.DataFrame,
    prediction: pd.DataFrame,
    output: Path | str | None = None,
    *,
    ax: Axes | None = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    title: str | None = None,
) -> Axes:
    """Single-method CNA heatmap.

    Parameters
    ----------
    cna
        ``(n_bins, n_cells)`` DataFrame. If its index is a ``MultiIndex`` whose
        first level is chromosome, a chromosome-colour strip is drawn above the
        heatmap.
    prediction
        DataFrame with columns ``cell`` (or ``cell.names``) and ``copykat.pred``;
        used to sort rows.
    output
        If given, save the figure to this PNG path. When ``ax`` is also given,
        ``output`` is ignored.
    ax
        Optional axes to draw into. When ``None``, a new figure is created. To
        save: pass ``output=...`` and leave ``ax=None``.

    Returns
    -------
    The heatmap axes.
    """
    col_order = _order_cells_by_pred(prediction, list(cna.columns))
    M = cna[col_order].to_numpy().T  # cells × bins
    chrom = _chrom_strip(cna.index)

    if ax is None:
        fig, (ax_cb, ax_heat) = plt.subplots(
            2, 1, figsize=(12, 10),
            gridspec_kw={"height_ratios": [0.5, 20], "hspace": 0.02},
        )
        _draw_chrom_bar(ax_cb, chrom)
        ax_cb.set_title(title or "chromosome", fontsize=9, pad=4)
        target = ax_heat
        owns_fig = True
    else:
        target = ax
        owns_fig = False
        if title is not None:
            target.set_title(title)

    target.imshow(
        M, aspect="auto", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
    )
    target.set_ylabel(f"{M.shape[0]} cells (sorted by pred)")
    target.set_xlabel(f"{M.shape[1]} bins")
    target.set_yticks([])
    target.set_xticks([])

    if owns_fig and output is not None:
        plt.gcf().savefig(Path(output), dpi=150, bbox_inches="tight")
        plt.close(plt.gcf())
    return target


def plot_cna_heatmap_compare(
    py_cna: pd.DataFrame,
    r_cna: pd.DataFrame,
    py_prediction: pd.DataFrame,
    r_prediction: pd.DataFrame,
    *,
    axes: tuple[Axes, Axes] | None = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    sort_by: str = "py",
) -> tuple[Axes, Axes]:
    """Side-by-side py vs R CNA heatmap on the **same cell ordering**.

    Both panels are sorted by the prediction selected by ``sort_by`` so visual
    differences are bin-level, not row-permutation. Cells present on one side
    but not the other are dropped.

    Parameters
    ----------
    py_cna, r_cna
        ``(n_bins, n_cells)`` DataFrames. Bin axis must align (same row index);
        cell axis is intersected.
    py_prediction, r_prediction
        Per-cell prediction tables (``cell`` or ``cell.names`` column +
        ``copykat.pred``).
    axes
        Optional pair of pre-created axes for the two panels. When ``None``, a
        new ``(1, 2)`` figure is created.
    sort_by
        ``"py"`` (default) or ``"r"`` — which side's labels drive the row order.
    """
    if sort_by not in ("py", "r"):
        raise ValueError(f"sort_by must be 'py' or 'r', got {sort_by!r}")
    if not py_cna.index.equals(r_cna.index):
        raise ValueError("py_cna and r_cna must share the same bin index")

    common = [c for c in py_cna.columns if c in r_cna.columns]
    if not common:
        raise ValueError("py_cna and r_cna share no cell columns")
    pred_for_sort = py_prediction if sort_by == "py" else r_prediction
    col_order = _order_cells_by_pred(pred_for_sort, common)

    M_py = py_cna[col_order].to_numpy().T
    M_r = r_cna[col_order].to_numpy().T
    chrom = _chrom_strip(py_cna.index)

    if axes is None:
        fig, (ax_py, ax_r) = plt.subplots(
            1, 2, figsize=(16, 8), sharey=True,
        )
    else:
        ax_py, ax_r = axes

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    for target, M, label in ((ax_py, M_py, "pycopykat"), (ax_r, M_r, "R copykat")):
        target.imshow(M, aspect="auto", cmap="RdBu_r", norm=norm)
        target.set_title(label)
        target.set_xlabel(f"{M.shape[1]} bins")
        target.set_yticks([])
        target.set_xticks([])
        # thin chromosome strip drawn at the top inside each panel
        if (chrom != 0).any():
            twin = target.inset_axes([0, 1.005, 1, 0.025])
            _draw_chrom_bar(twin, chrom)
    ax_py.set_ylabel(f"{M_py.shape[0]} cells (sort_by={sort_by})")

    return ax_py, ax_r


def plot_cna_delta(
    py_cna: pd.DataFrame,
    r_cna: pd.DataFrame,
    prediction: pd.DataFrame | None = None,
    *,
    ax: Axes | None = None,
    vmin: float = -0.5,
    vmax: float = 0.5,
) -> Axes:
    """Heatmap of bin-level ``py - R`` differences over the common cell set.

    Parameters
    ----------
    py_cna, r_cna
        ``(n_bins, n_cells)`` DataFrames; bin index must align, cell axis is
        intersected.
    prediction
        Optional prediction table to drive cell ordering. When ``None``, cells
        are ordered alphabetically.
    ax
        Optional axes; when ``None``, a new figure is created.
    """
    if not py_cna.index.equals(r_cna.index):
        raise ValueError("py_cna and r_cna must share the same bin index")
    common = [c for c in py_cna.columns if c in r_cna.columns]
    if not common:
        raise ValueError("py_cna and r_cna share no cell columns")
    if prediction is not None:
        col_order = _order_cells_by_pred(prediction, common)
    else:
        col_order = sorted(common)
    delta = (py_cna[col_order].to_numpy() - r_cna[col_order].to_numpy()).T  # cells × bins

    if ax is None:
        _fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(
        delta, aspect="auto", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
    )
    ax.set_title("Δ CNA  (pycopykat − R copykat)")
    ax.set_xlabel(f"{delta.shape[1]} bins")
    ax.set_ylabel(f"{delta.shape[0]} cells")
    ax.set_yticks([])
    ax.set_xticks([])
    return ax
