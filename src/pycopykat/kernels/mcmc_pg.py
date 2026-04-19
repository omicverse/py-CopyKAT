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
