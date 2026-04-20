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

    # Assign each gene to a bin via per-chromosome searchsorted on the bin-end
    # boundaries. Genes whose center <= 0 (start) of the first bin or > the
    # last bin's end on that chrom, and genes whose chrom has no bins, get -1
    # and are excluded from the groupby (matches the old mask-based behaviour:
    # the loop only produced a non-NaN median for bins whose mask was true).
    bin_id = np.full(gene_chr.size, -1, dtype=np.int64)
    # Global-index anchor per unique bin_chr (bins are in per-chromosome order)
    for chrom in np.unique(bin_chr):
        bmask = bin_chr == chrom
        gmask = gene_chr == chrom
        if not bmask.any() or not gmask.any():
            continue
        ends = bin_end[bmask]
        starts = bin_start[bmask]
        global_idx = np.flatnonzero(bmask)
        c = centers[gmask]
        # "center <= end" → searchsorted(ends, c, side="left")
        idx = np.searchsorted(ends, c, side="left")
        valid = idx < ends.size  # genes past the last bin: excluded (old code
                                 # had no fall-through either — their mask
                                 # never matched any bin on this chrom)
        # R copykat's window is (start, end]: also require c > starts[idx]
        idx_clamped = np.where(valid, idx, 0)
        sel = starts[idx_clamped]
        valid_win = valid & (c > sel)
        assigned = np.where(valid_win, global_idx[idx_clamped], -1)
        out_idx = np.flatnonzero(gmask)
        bin_id[out_idx] = assigned

    # Per-bin median: sort gene rows by bin id so each bin's genes form a
    # contiguous run. Single-gene bins (majority) are copied directly; bins
    # with ≥ 2 genes get one np.median call each. Empty bins stay NaN and get
    # filled below.
    keep = bin_id >= 0
    medians = np.full((n_bins, n_cells), np.nan, dtype=np.float64)
    if keep.any():
        g_ids = bin_id[keep]
        g_vals = logCNA[keep]
        order = np.argsort(g_ids, kind="stable")
        g_ids_s = g_ids[order]
        g_vals_s = g_vals[order]
        # Run boundaries in the sorted array
        changes = np.flatnonzero(np.diff(g_ids_s)) + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [g_ids_s.size]))
        run_bin = g_ids_s[starts]
        run_len = ends - starts
        # Size-1 runs: copy directly
        one_mask = run_len == 1
        if one_mask.any():
            medians[run_bin[one_mask]] = g_vals_s[starts[one_mask]]
        # Size-2 runs: mean (cheaper than median for n=2 and equal)
        two_mask = run_len == 2
        if two_mask.any():
            s2 = starts[two_mask]
            medians[run_bin[two_mask]] = 0.5 * (g_vals_s[s2] + g_vals_s[s2 + 1])
        # Size >= 3: explicit median over the small slice
        big = np.flatnonzero(run_len >= 3)
        for k in big:
            medians[run_bin[k]] = np.median(
                g_vals_s[starts[k]:ends[k]], axis=0
            )

    # Forward fill: replace NaN bins with the most recent non-NaN per cell.
    # The Python-level loop over n_bins is fine — each iteration is a single
    # boolean-indexed assignment over n_cells entries, which stays in C.
    out = medians
    last = np.full(n_cells, np.nan, dtype=np.float64)
    for b in range(n_bins):
        missing = np.isnan(out[b])
        out[b, missing] = last[missing]
        present = ~missing
        last[present] = out[b, present]
    # Back fill any remaining leading NaNs.
    nxt = np.full(n_cells, np.nan, dtype=np.float64)
    for b in range(n_bins - 1, -1, -1):
        missing = np.isnan(out[b])
        out[b, missing] = nxt[missing]
        present = ~missing
        nxt[present] = out[b, present]
    return out
