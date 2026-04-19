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


def vst_center(x: NDArray[np.floating] | NDArray[np.integer]) -> NDArray[np.float64]:
    """Apply ``log(sqrt(x) + sqrt(x+1))`` and center each column at zero.

    Parameters
    ----------
    x
        Genes × cells count (or transformed count) array, any numeric dtype.

    Returns
    -------
    np.ndarray (float64)
        Same shape as ``x``. Each column has mean ≈ 0.
    """
    y = np.asarray(x, dtype=np.float64)
    y = np.log(np.sqrt(y) + np.sqrt(y + 1.0))
    y -= y.mean(axis=0, keepdims=True)
    return y
