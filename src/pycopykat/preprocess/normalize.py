"""Variance-stabilizing transform and per-cell centering (copykat.R ~L105-107).

R reference::

    norm.mat <- log(sqrt(x) + sqrt(x+1))
    norm.mat <- apply(norm.mat, 2, function(x) x - mean(x))

This is the Anscombe-like Freeman–Tukey VST followed by per-column (per-cell)
mean centering.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


def vst_center(
    x: "NDArray[np.floating] | NDArray[np.integer] | sparse.spmatrix",
) -> NDArray[np.float64]:
    """Apply ``log(sqrt(x) + sqrt(x+1))`` and center each column at zero.

    Accepts a dense ndarray OR a ``scipy.sparse`` matrix. The VST
    ``f(x) = log(sqrt(x) + sqrt(x+1))`` maps ``0 → log(1) = 0`` so it
    preserves sparsity and only touches ``.data`` on sparse input. Centering
    subtracts the per-column (per-cell) mean and densifies — that is the
    sparse→dense hand-off boundary. The caller never sees a sparse return.

    Parameters
    ----------
    x
        Genes × cells count (or transformed count) array; dense or sparse.

    Returns
    -------
    np.ndarray (float64)
        Dense, same shape as ``x``. Each column has mean ≈ 0.
    """
    if sparse.issparse(x):
        # ``astype(copy=True)`` copies ``.data`` (and coerces to float64) while
        # preserving sparsity structure; ``indices`` / ``indptr`` are shared but
        # never mutated below, so the caller's matrix stays untouched.
        y = x.astype(np.float64, copy=True)
        y.data = np.log(np.sqrt(y.data) + np.sqrt(y.data + 1.0))
        # Zeros contribute nothing to the column sum, so summing ``.data`` via
        # ``sum(axis=0)`` over implicit zeros is O(nnz) and numerically exact
        # for the VST-of-zero-equals-zero branch. ``sum`` returns np.matrix
        # (1, n_cols); flatten to 1-D ndarray for broadcasting.
        col_means = np.asarray(y.sum(axis=0)).ravel() / y.shape[0]
        y_dense = y.toarray()
        y_dense -= col_means[None, :]
        return y_dense

    # dense path — unchanged (bit-identical to pre-S2)
    y = np.asarray(x, dtype=np.float64)
    y = np.log(np.sqrt(y) + np.sqrt(y + 1.0))
    y -= y.mean(axis=0, keepdims=True)
    return y
