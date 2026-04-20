"""Pairwise distance kernels. All take (n, p) arrays and return condensed
distance vectors of length n*(n-1)/2 (scipy convention)."""
# Uppercase `X` / `R` match the matrix-data convention used by scipy/sklearn.
# ruff: noqa: N803, N806
from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy.spatial.distance import pdist

# Below this, scipy's single-threaded C kernel already beats BLAS setup cost
# and avoids catastrophic cancellation for near-identical rows.
_BLAS_EUCLIDEAN_MIN_N = 100


def pdist_euclidean(X: np.ndarray) -> np.ndarray:
    """BLAS-based condensed euclidean distance vector.

    Uses ``||a - b||^2 = ||a||^2 + ||b||^2 - 2*a*b^T`` via a single GEMM.
    Parallel when NumPy is linked against a multi-threaded BLAS
    (OpenBLAS/MKL). Output matches ``scipy.spatial.distance.pdist(X,
    "euclidean")`` to numerical precision (atol ~1e-8); the
    symmetric-identity formula can produce tiny negative squared
    distances from catastrophic cancellation for near-identical rows,
    which we clamp to zero before the sqrt.

    For ``n < 100`` rows we delegate to scipy directly: its C kernel
    already beats the BLAS GEMM setup at that size and avoids the
    cancellation issue entirely.

    Returns the upper-triangle (k=1) in row-major order — identical
    ordering to ``scipy.spatial.distance.pdist``.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < _BLAS_EUCLIDEAN_MIN_N:
        return pdist(X, metric="euclidean")

    # Squared norms per row — use einsum to fuse square + sum.
    sq = np.einsum("ij,ij->i", X, X)
    # Level-3 BLAS GEMM — this is the parallel hot step on OpenBLAS/MKL.
    G = X @ X.T
    # Broadcast outer sum minus 2·Gram → squared distances.
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    # Clamp negatives (catastrophic cancellation for near-identical rows).
    np.maximum(D2, 0.0, out=D2)
    # Extract upper triangle (k=1) in row-major order — matches scipy.pdist.
    iu, ju = np.triu_indices(n, k=1)
    return np.sqrt(D2[iu, ju])


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
