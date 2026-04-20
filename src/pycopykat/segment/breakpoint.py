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
from scipy.stats import gamma, ks_2samp

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


def ks_gamma_analytic(
    shape1: float,
    scale1: float,
    shape2: float,
    scale2: float,
    *,
    n_grid: int = 4096,
) -> float:
    """Analytic Kolmogorov-Smirnov statistic between two Gamma distributions.

    Computes ``sup_x |F1(x) - F2(x)|`` where ``F_i`` is the CDF of
    ``Gamma(shape_i, scale=scale_i)``. The supremum is approximated on a
    dense ``n_grid``-point linear grid spanning the union support (from the
    min-tail 1e-4 quantile to the max-tail 0.9999 quantile across both
    distributions). Because Gamma CDFs are smooth, 4096 points are more than
    enough to pin down the sup to well below the MC KS noise floor at
    ``mc=1000`` (~0.04).

    Parameters
    ----------
    shape1, scale1, shape2, scale2
        Shape and scale parameters of the two Gamma distributions.
    n_grid
        Grid resolution; default 4096.

    Returns
    -------
    float
        The analytic KS statistic in ``[0, 1]``.

    Notes
    -----
    Uses ``scipy.stats.gamma.cdf`` (vectorized) — no Python loop over the
    grid. For the Poisson-Gamma posterior here, shapes are
    ``α_i + Σy_i`` with ``α_i = max(mean(y_i), 1e-3)`` and scales are
    ``1 / (β + n_i) = 1 / (1 + n_i)``.
    """
    s1 = float(shape1)
    s2 = float(shape2)
    sc1 = float(scale1)
    sc2 = float(scale2)
    # Union-support grid. ppf is the exact Gamma quantile — scipy handles
    # the near-zero-shape degenerate case (mass concentrated near 0) without
    # NaNs.
    x_lo = float(min(gamma.ppf(1e-4, s1, scale=sc1), gamma.ppf(1e-4, s2, scale=sc2)))
    x_hi = float(
        max(gamma.ppf(0.9999, s1, scale=sc1), gamma.ppf(0.9999, s2, scale=sc2))
    )
    if not np.isfinite(x_lo):
        x_lo = 0.0
    if not np.isfinite(x_hi) or x_hi <= x_lo:
        # Both distributions collapsed to a point; KS is 0 if shapes/scales
        # match, 1 otherwise.
        return 0.0 if (s1 == s2 and sc1 == sc2) else 1.0
    grid = np.linspace(x_lo, x_hi, int(n_grid))
    cdf1 = gamma.cdf(grid, s1, scale=sc1)
    cdf2 = gamma.cdf(grid, s2, scale=sc2)
    return float(np.abs(cdf1 - cdf2).max())


def find_breakpoints_analytic(
    y: NDArray[np.floating],
    *,
    bins: int,
    ks_cut: float,
    n_grid: int = 4096,
) -> list[int]:
    """Breakpoint detection using the closed-form Gamma-KS statistic.

    Same window-pair loop as :func:`find_breakpoints` but replaces the
    Monte-Carlo ``ks_2samp`` call with :func:`ks_gamma_analytic`. Produces a
    deterministic result independent of any RNG seed.

    Parameters
    ----------
    y
        Length-``n`` genomic signal.
    bins
        Window width in genes.
    ks_cut
        KS statistic threshold for calling a breakpoint. ``ks_cut=0.1`` was
        tuned against the MC noise floor at ``mc=1000``; the analytic path
        has no MC noise, so the effective threshold differs and may need
        retuning.
    n_grid
        Grid resolution passed through to :func:`ks_gamma_analytic`.

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
        # Posterior Gamma(α + Σy, scale=1/(β+n)); β=1 per pg_posterior_samples.
        shape1 = a1 + float(y1.sum())
        shape2 = a2 + float(y2.sum())
        scale1 = 1.0 / (1.0 + y1.size)
        scale2 = 1.0 / (1.0 + y2.size)
        stat = ks_gamma_analytic(
            shape1, scale1, shape2, scale2, n_grid=n_grid
        )
        if stat > ks_cut:
            breaks.append(boundaries[i + 1])

    return sorted({0, *breaks, n - 1})
