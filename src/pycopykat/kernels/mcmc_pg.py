"""Poisson-Gamma conjugate posterior sampling.

MCMCpack::MCpoissongamma(y, alpha, beta, mc) = rgamma(mc, alpha+sum(y), beta+n).
R's rgamma(shape, rate) equals numpy.random.Generator.gamma(shape, scale=1/rate).
"""
from __future__ import annotations

import numpy as np
from numba import njit


def pg_posterior_samples(
    y: np.ndarray, alpha: float, beta: float, mc: int, seed: int
) -> np.ndarray:
    """Draw ``mc`` i.i.d. samples from ``Gamma(alpha + sum(y), rate=beta + n)``.

    This is the exact conjugate posterior of a Poisson rate parameter under a
    ``Gamma(alpha, beta)`` prior and observations ``y``. Matches the behaviour
    of ``MCMCpack::MCpoissongamma`` in R (which is itself a pure ``rgamma``
    draw — no Metropolis-Hastings, no burn-in).

    Parameters
    ----------
    y : np.ndarray
        Observed non-negative integer counts (stored as float64 for
        compatibility with the rest of the pipeline).
    alpha, beta : float
        Gamma prior hyper-parameters (shape, rate).
    mc : int
        Number of Monte-Carlo draws to return.
    seed : int
        Seed for a fresh ``numpy.random.default_rng`` instance.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(mc,)`` with posterior draws for lambda.
    """
    rng = np.random.default_rng(seed)
    shape = alpha + float(y.sum())
    scale = 1.0 / (beta + y.size)
    return rng.gamma(shape=shape, scale=scale, size=mc)


def pg_posterior_mean(
    y: np.ndarray, alpha: float, beta: float, mc: int, seed: int
) -> float:
    """Monte-Carlo estimate of the posterior mean of lambda.

    Convenience wrapper around :func:`pg_posterior_samples`. The analytic
    posterior mean is ``(alpha + sum(y)) / (beta + n)``; this function returns
    the empirical mean of ``mc`` draws, which converges to that value.
    """
    return float(pg_posterior_samples(y, alpha, beta, mc, seed).mean())


@njit(cache=True, fastmath=True)
def pg_posterior_mean_from_shape_scale(
    shape: float, scale: float, draws: np.ndarray
) -> float:
    """Aggregate pre-sampled Gamma draws into a Monte-Carlo posterior mean.

    This helper does **not** sample — it assumes ``draws`` already holds
    Gamma(shape, scale=1) variates (produced by NumPy outside the JIT region,
    since Numba's RNG support is limited). It simply multiplies by ``scale``
    and averages. The ``shape`` argument is accepted for API symmetry with a
    possible future in-kernel sampler; it is currently unused.

    Reserved for Task 5.2, where the speed-critical outer loop will be
    parallelised via ``joblib`` and per-segment aggregation benefits from a
    Numba-compiled reducer.
    """
    # ``shape`` is intentionally unused in the V1 aggregator; keep the
    # reference so the JIT sees a consistent signature.
    _ = shape
    n = draws.shape[0]
    total = 0.0
    for i in range(n):
        total += draws[i] * scale
    return total / n
