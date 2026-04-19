"""Pairwise distance kernels. All take (n, p) arrays and return condensed
distance vectors of length n*(n-1)/2 (scipy convention)."""
# Uppercase `X` / `R` match the matrix-data convention used by scipy/sklearn.
# ruff: noqa: N803, N806
from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy.spatial.distance import pdist


def pdist_euclidean(X: np.ndarray) -> np.ndarray:
    """Thin wrapper over scipy (already C-optimized)."""
    return pdist(np.ascontiguousarray(X, dtype=np.float64), metric="euclidean")


@njit(cache=True, parallel=True, fastmath=True)
def _pdist_pearson_kernel(X: np.ndarray) -> np.ndarray:
    n, p = X.shape
    # Center rows
    means = np.empty(n)
    stds = np.empty(n)
    for i in prange(n):
        m = 0.0
        for k in range(p):
            m += X[i, k]
        m /= p
        means[i] = m
        s = 0.0
        for k in range(p):
            d = X[i, k] - m
            s += d * d
        stds[i] = np.sqrt(s / p)
    out = np.empty(n * (n - 1) // 2)
    # Fill (i,j) pairs. Compute per-row index offsets.
    # cumulative pairs before row i: i*(2n - i - 1) / 2
    for i in prange(n - 1):
        offset = i * (2 * n - i - 1) // 2
        for j in range(i + 1, n):
            mi, mj = means[i], means[j]
            si, sj = stds[i], stds[j]
            if si == 0.0 or sj == 0.0:
                out[offset + (j - i - 1)] = 1.0
                continue
            dot = 0.0
            for k in range(p):
                dot += (X[i, k] - mi) * (X[j, k] - mj)
            r = dot / (p * si * sj)
            out[offset + (j - i - 1)] = 1.0 - r
    return out


def pdist_pearson(X: np.ndarray) -> np.ndarray:
    return _pdist_pearson_kernel(np.ascontiguousarray(X, dtype=np.float64))


def pdist_spearman(X: np.ndarray) -> np.ndarray:
    """Rank-transform rows, then delegate to Pearson."""
    from scipy.stats import rankdata

    R = np.apply_along_axis(rankdata, 1, X)
    return pdist_pearson(np.ascontiguousarray(R, dtype=np.float64))
