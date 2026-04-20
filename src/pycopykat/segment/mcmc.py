"""Per-cell Poisson-Gamma segmentation with a shared breakpoint set.

Given a cells-with-cluster-labels matrix of (VST, centered, smoothed) gene
expression, :func:`segment_cells`:

1. Computes per-cluster consensus (gene-wise median) signals.
2. Takes the **union** of breakpoints detected on each consensus signal.
3. For every segment of the shared breakpoint set, computes the analytical
   Poisson-Gamma posterior mean of each cell's expression on that segment
   (``(α + Σy) / (1 + n_seg)``) in one :func:`numpy.add.reduceat` call, then
   broadcasts the ``(n_seg, n_cells)`` result back into ``(n_genes, n_cells)``
   via :func:`numpy.repeat`.
4. Returns ``log`` of the piecewise-constant matrix (``logCNA``) plus the
   breakpoint list.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pycopykat.segment.breakpoint import (
    find_breakpoints,
    find_breakpoints_analytic,
)


def _consensus_per_cluster(
    fttmat: NDArray[np.floating], clu: NDArray[np.integer]
) -> NDArray[np.float64]:
    """Per-cluster gene-wise median, then ``exp`` (matches R ``exp(CON)``)."""
    parts: list[NDArray[np.float64]] = []
    for lab in np.unique(clu):
        med = np.median(fttmat[:, clu == lab], axis=1)
        parts.append(np.exp(med).astype(np.float64))
    return np.column_stack(parts)


def _union_breaks(
    consensus: NDArray[np.floating],
    *,
    bins: int,
    ks_cut: float,
    seed: int,
    mc: int,
    breakpoint_ks: str = "mc",
) -> list[int]:
    """Union of per-cluster breakpoint calls.

    ``breakpoint_ks="mc"`` (default) runs the MC Poisson-Gamma KS path,
    bit-identical to the original behaviour. ``breakpoint_ks="analytic"``
    dispatches to :func:`find_breakpoints_analytic`, which uses a closed-form
    Gamma-KS statistic and is deterministic (ignores ``seed`` and ``mc``).
    """
    all_breaks: set[int] = {0, consensus.shape[0] - 1}
    for c in range(consensus.shape[1]):
        if breakpoint_ks == "analytic":
            br = find_breakpoints_analytic(
                consensus[:, c], bins=bins, ks_cut=ks_cut,
            )
        else:
            br = find_breakpoints(
                consensus[:, c], bins=bins, ks_cut=ks_cut,
                seed=seed + 1000 * c, mc=mc,
            )
        all_breaks.update(br)
    return sorted(all_breaks)


def segment_cells(
    fttmat: NDArray[np.floating],
    clu: NDArray[np.integer],
    *,
    bins: int,
    ks_cut: float,
    seed: int,
    mc: int = 1000,
    n_jobs: int = 1,
    breakpoint_ks: str = "mc",
) -> tuple[NDArray[np.float64], list[int]]:
    """Per-cell analytic Poisson-Gamma segmentation with cluster-union breaks.

    Parameters
    ----------
    fttmat
        ``(n_genes, n_cells)`` VST-smoothed relative expression (pre-``exp``).
    clu
        Length-``n_cells`` 1-based cluster labels.
    bins
        Window width in genes for breakpoint detection.
    ks_cut
        KS statistic threshold for calling a breakpoint.
    seed
        Base seed for KS breakpoint detection on the consensus signals.
    mc
        Posterior samples per window in the KS-breakpoint step.
        (Not used for segment means — those are closed-form.)
    n_jobs
        Unused; kept for backwards-compatible signature. The vectorized
        segmentation runs in a single NumPy call, so per-cell parallelism
        is not needed here.
    breakpoint_ks
        ``"mc"`` (default) uses the Monte-Carlo Poisson-Gamma KS test
        (``mc`` samples per window). ``"analytic"`` uses the closed-form
        Gamma-KS statistic from :func:`pycopykat.segment.breakpoint.find_breakpoints_analytic`
        (experimental; ``ks_cut`` may need retuning).

    Returns
    -------
    (logCNA, BR)
        ``logCNA`` is ``(n_genes, n_cells)`` piecewise-constant log-expression.
        ``BR`` is the sorted shared breakpoint list.
    """
    del n_jobs  # kept for API compat; vectorized path has no per-cell loop
    consensus = _consensus_per_cluster(fttmat, clu)
    BR = _union_breaks(
        consensus,
        bins=bins,
        ks_cut=ks_cut,
        seed=seed,
        mc=mc,
        breakpoint_ks=breakpoint_ks,
    )

    raw = np.exp(np.asarray(fttmat, dtype=np.float64))
    n_genes = raw.shape[0]

    # Segment index construction (shared across all cells).
    # _segment_one_cell semantics (now removed): seg 0 covers [BR[0], BR[1]]
    # inclusive; seg i>=1 covers [BR[i]+1, BR[i+1]] inclusive. Equivalently,
    # seg_starts = [BR[0], BR[1]+1, ..., BR[K-2]+1]; each segment ends at
    # the next start minus one, or at n_genes for the last segment.
    br = np.asarray(BR, dtype=np.int64)
    seg_starts = np.concatenate(([br[0]], br[1:-1] + 1))  # length K-1
    assert seg_starts.size == len(BR) - 1

    seg_lens = np.diff(
        np.concatenate([seg_starts, [n_genes]])
    ).astype(np.float64)
    assert int(seg_lens.sum()) == n_genes

    seg_sums = np.add.reduceat(raw, seg_starts, axis=0)  # (n_seg, n_cells)
    seg_means = seg_sums / seg_lens[:, None]
    a = np.maximum(seg_means, 1e-3)
    post_mean = (a + seg_sums) / (1.0 + seg_lens[:, None])  # (n_seg, n_cells)

    seg_ids = np.repeat(np.arange(seg_starts.size), seg_lens.astype(np.int64))
    logCNA = np.log(np.maximum(post_mean[seg_ids], 1e-12))
    return logCNA, BR
