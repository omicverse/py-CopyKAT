"""Breakpoint detection via KS test on Poisson-Gamma posterior samples.

R copykat walks a fixed-width window (``bins`` genes) across each cluster's
consensus expression and, for every adjacent pair of windows, compares their
Poisson-Gamma posterior predictive distributions via a 2-sample KS statistic.
A boundary is retained when the statistic exceeds ``ks_cut``. The endpoints
(0 and n-1) are always included in the returned list.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp

from pycopykat.kernels.mcmc_pg import pg_posterior_samples


def find_breakpoints(
    y: NDArray[np.floating],
    *,
    bins: int,
    ks_cut: float,
    seed: int,
    mc: int = 1000,
) -> list[int]:
    """Locate breakpoints on a single 1-D genomic signal.

    Parameters
    ----------
    y
        Length-``n`` genomic signal (typically per-cluster consensus expression).
    bins
        Window width in genes.
    ks_cut
        Threshold on the KS statistic above which a window boundary counts
        as a breakpoint.
    seed
        Base seed; each comparison uses ``seed + i`` and ``seed + i + 10_000``
        for left/right posterior draws.
    mc
        Posterior samples per window.

    Returns
    -------
    Sorted list of breakpoint indices, always including ``0`` and ``n-1``.
    """
    n = int(y.size)
    if n < 3 * bins:
        return [0, n - 1]

    boundaries = list(range(0, (n // bins - 1) * bins + 1, bins))
    if boundaries[-1] != n - 1:
        boundaries.append(n - 1)

    breaks: list[int] = []
    for i in range(len(boundaries) - 2):
        s1, e1 = boundaries[i], boundaries[i + 1]
        s2, e2 = boundaries[i + 1] + 1, boundaries[i + 2]
        y1 = y[s1:e1]
        y2 = y[s2:e2]
        if y1.size == 0 or y2.size == 0:
            continue
        a1 = max(float(y1.mean()), 1e-3)
        a2 = max(float(y2.mean()), 1e-3)
        p1 = pg_posterior_samples(y1, alpha=a1, beta=1.0, mc=mc, seed=seed + i)
        p2 = pg_posterior_samples(
            y2, alpha=a2, beta=1.0, mc=mc, seed=seed + i + 10_000
        )
        stat, _ = ks_2samp(p1, p2)
        if float(stat) > ks_cut:
            breaks.append(boundaries[i + 1])

    return sorted({0, *breaks, n - 1})
