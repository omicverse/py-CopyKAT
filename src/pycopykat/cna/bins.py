"""Gene-level → 220 kb genomic bin aggregation (R ``convert.all.bins.hg20``).

The shipped bin table (``data/hg20_220kb_bins.parquet``) stores a single
``chrompos`` per bin representing the bin's *end* coordinate (within its
chromosome). The bin's start is the previous bin's ``chrompos`` on the same
chromosome; first bin on each chromosome starts at 0. For every bin we take
the per-cell median across all genes whose coordinate centre falls within
``(start, end]`` on the matching chromosome. Bins with no resident gene are
forward-filled along the genome (and leading NaNs back-filled).

Matches R copykat's ``convert.all.bins.hg20``: drops chromosome 24 (Y) by
default, per-bin median aggregation, fill-in for empty bins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def aggregate_to_bins(
    logCNA: NDArray[np.floating],
    gene_anno: pd.DataFrame,
    bins: pd.DataFrame,
    *,
    exclude_chrom_24: bool = True,
) -> NDArray[np.float64]:
    """Aggregate a genes × cells matrix down to 220 kb bins × cells.

    Parameters
    ----------
    logCNA
        ``(n_genes, n_cells)`` matrix whose rows are aligned with ``gene_anno``.
    gene_anno
        DataFrame with at least ``chromosome_name``, ``start_position``,
        ``end_position`` columns.
    bins
        DataFrame with at least ``chrom`` and ``chrompos`` columns (the
        ``hg20_220kb_bins.parquet`` schema). ``chrompos`` is the bin end.
    exclude_chrom_24
        Drop chromosome 24 (Y) from ``bins`` before aggregation; matches R.

    Returns
    -------
    ``(n_retained_bins, n_cells)`` array of per-bin medians with empty bins
    filled forward then (any remaining) back-filled.
    """
    if exclude_chrom_24:
        bins = bins.loc[bins["chrom"] != 24].reset_index(drop=True)
    else:
        bins = bins.reset_index(drop=True)
    bins = bins.copy()
    bins["_end"] = bins["chrompos"].astype(np.float64)
    bins["_start"] = (
        bins.groupby("chrom", sort=False)["_end"].shift(1).fillna(0.0).astype(np.float64)
    )

    centers = 0.5 * (
        gene_anno["start_position"].to_numpy(np.float64)
        + gene_anno["end_position"].to_numpy(np.float64)
    )
    gene_chr = gene_anno["chromosome_name"].to_numpy(np.int64)
    bin_chr = bins["chrom"].to_numpy(np.int64)
    bin_start = bins["_start"].to_numpy()
    bin_end = bins["_end"].to_numpy()

    logCNA = np.asarray(logCNA, dtype=np.float64)
    n_bins = len(bins)
    n_cells = logCNA.shape[1]
    out = np.full((n_bins, n_cells), np.nan, dtype=np.float64)

    for b in range(n_bins):
        mask = (
            (gene_chr == bin_chr[b])
            & (centers > bin_start[b])
            & (centers <= bin_end[b])
        )
        if not mask.any():
            continue
        out[b] = np.median(logCNA[mask], axis=0)

    # Forward fill: replace NaN bins with the most recent non-NaN per cell
    last = np.full(n_cells, np.nan)
    for b in range(n_bins):
        missing = np.isnan(out[b])
        out[b, missing] = last[missing]
        present = ~np.isnan(out[b])
        last[present] = out[b, present]

    # Back fill any remaining leading NaNs
    nxt = np.full(n_cells, np.nan)
    for b in range(n_bins - 1, -1, -1):
        missing = np.isnan(out[b])
        out[b, missing] = nxt[missing]
        present = ~np.isnan(out[b])
        nxt[present] = out[b, present]

    return out
