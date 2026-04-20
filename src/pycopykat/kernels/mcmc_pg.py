"""Poisson-Gamma conjugate posterior sampling.

MCMCpack::MCpoissongamma(y, alpha, beta, mc) = rgamma(mc, alpha+sum(y), beta+n).
R's rgamma(shape, rate) equals numpy.random.Generator.gamma(shape, scale=1/rate).
"""
from __future__ import annotations

import numpy as np


def pg_posterior_samples(
    y: np.ndarray, alpha: float, beta: float, mc: int, seed: int
) -> np.ndarray:
    """Draw `mc` i.i.d. samples from Gamma(alpha + sum(y), rate=beta + n)."""
    rng = np.random.default_rng(seed)
    shape = alpha + float(y.sum())
    scale = 1.0 / (beta + y.size)
    return rng.gamma(shape=shape, scale=scale, size=mc)


def pg_posterior_mean(
    y: np.ndarray, alpha: float, beta: float, mc: int, seed: int
) -> float:
    """Monte-Carlo estimate of posterior mean lambda."""
    return float(pg_posterior_samples(y, alpha, beta, mc, seed).mean())


def pg_posterior_mean_analytic(
    y: np.ndarray, alpha: float, beta: float
) -> float:
    """Analytical posterior mean of ``Gamma(alpha + sum(y), scale=1/(beta + n))``.

    Gamma(shape=α+Σy, scale=1/(β+n)) has expectation ``shape · scale``,
    which equals ``(α + Σy) / (β + n)``. Substituting this closed form for
    a ``mc``-sample Monte-Carlo mean removes ~``1/√mc`` noise and avoids an
    RNG call, without changing the expectation.
    """
    return float((alpha + float(y.sum())) / (beta + y.size))
