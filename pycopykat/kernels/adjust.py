"""Per-cell threshold adjustment (R copykat adjN).

For each cell column, mutes bin values that are within factor*sd of the
baseline to the cell's own mean. Matches R copykat.R lines 378-391.
"""
# ruff: noqa: N803, N806  # uppercase X matrix name by convention
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=True)
def _adjust_kernel(X: np.ndarray, base: np.ndarray, sd: np.ndarray, factor: float) -> np.ndarray:
    g, c = X.shape
    out = np.empty_like(X)
    for j in prange(c):
        col_mean = 0.0
        for i in range(g):
            col_mean += X[i, j]
        col_mean /= g
        for i in range(g):
            if abs(X[i, j] - base[i]) < factor * sd[i]:
                out[i, j] = col_mean
            else:
                out[i, j] = X[i, j]
    return out


def adjust_threshold(
    X: np.ndarray, base: np.ndarray, sd: np.ndarray, factor: float = 0.25
) -> np.ndarray:
    """Public wrapper. Coerces inputs to contiguous float64 before JIT kernel."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    base = np.ascontiguousarray(base, dtype=np.float64)
    sd = np.ascontiguousarray(sd, dtype=np.float64)
    return _adjust_kernel(X, base, sd, float(factor))
