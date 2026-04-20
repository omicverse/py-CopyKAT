"""Per-cell Poisson-Gamma segmentation with a shared breakpoint set.

Given a cells-with-cluster-labels matrix of (VST, centered, smoothed) gene
expression, :func:`segment_cells`:

1. Computes per-cluster consensus (gene-wise median) signals.
2. Takes the **union** of breakpoints detected on each consensus signal.
3. For every cell independently, replaces each segment's values with the
   posterior mean of a Poisson-Gamma model conditioned on the segment's
   data (Numba kernel in :mod:`pycopykat.kernels.mcmc_pg`).
4. Returns ``log`` of the piecewise-constant matrix (``logCNA``) plus the
   breakpoint list.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray

from pycopykat.kernels.mcmc_pg import pg_posterior_mean_analytic
from pycopykat.segment.breakpoint import find_breakpoints


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
) -> list[int]:
    all_breaks: set[int] = {0, consensus.shape[0] - 1}
    for c in range(consensus.shape[1]):
        br = find_breakpoints(
            consensus[:, c], bins=bins, ks_cut=ks_cut,
            seed=seed + 1000 * c, mc=mc,
        )
        all_breaks.update(br)
    return sorted(all_breaks)


def _segment_one_cell(
    col: NDArray[np.floating], BR: list[int], seed: int, mc: int
) -> NDArray[np.float64]:
    x = np.empty_like(col, dtype=np.float64)
    # Segment i occupies [BR[i], BR[i+1]] (inclusive); subsequent segments start
    # at BR[i]+1 so boundary indices never get written twice.
    for i in range(len(BR) - 1):
        s = BR[i] if i == 0 else BR[i] + 1
        e = BR[i + 1]
        seg = col[s : e + 1]
        a = max(float(seg.mean()), 1e-3)
        # Analytical posterior mean of Gamma(a + Σseg, scale=1/(1 + n_seg)).
        # Replaces a 1000-sample MC mean with its closed-form expectation.
        x[s : e + 1] = pg_posterior_mean_analytic(seg, alpha=a, beta=1.0)
    return np.log(np.maximum(x, 1e-12))


def segment_cells(
    fttmat: NDArray[np.floating],
    clu: NDArray[np.integer],
    *,
    bins: int,
    ks_cut: float,
    seed: int,
    mc: int = 1000,
    n_jobs: int = 1,
) -> tuple[NDArray[np.float64], list[int]]:
    """Per-cell MCMC segmentation with cluster-union breakpoints.

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
        Base seed; per-cell seeds are ``seed + cell_index``.
    mc
        Posterior samples per segment in the mean estimate.
    n_jobs
        Parallel workers (joblib ``prefer="threads"``).

    Returns
    -------
    (logCNA, BR)
        ``logCNA`` is ``(n_genes, n_cells)`` piecewise-constant log-expression.
        ``BR`` is the sorted shared breakpoint list.
    """
    consensus = _consensus_per_cluster(fttmat, clu)
    BR = _union_breaks(consensus, bins=bins, ks_cut=ks_cut, seed=seed, mc=mc)

    raw = np.exp(np.asarray(fttmat, dtype=np.float64))
    cols = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_segment_one_cell)(raw[:, j], BR, seed + j, mc)
        for j in range(raw.shape[1])
    )
    logCNA = np.column_stack(cols).astype(np.float64)
    return logCNA, BR
