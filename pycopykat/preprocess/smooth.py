"""Per-cell Kalman smoothing followed by column re-centering.

R copykat step 3 applies a local-level DLM (``dlm::dlmSmooth``) to every
column independently, drops the initial prior sample, then subtracts the
per-column mean. The underlying smoother is in
:mod:`pycopykat.kernels.kalman`; this module supplies the post-centering.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pycopykat.kernels.kalman import kalman_smooth_matrix


def smooth_cells(
    X: NDArray[np.floating],
    dV: float = 0.16,
    dW: float = 0.001,
) -> NDArray[np.float64]:
    """Apply per-column Kalman smoothing then re-center each column.

    Parameters
    ----------
    X
        Genes × cells VST-centered matrix.
    dV, dW
        Observation and state-evolution variances for the local-level DLM.
        Defaults match R copykat (``dV = 0.16``, ``dW = 0.001``).

    Returns
    -------
    np.ndarray (float64)
        Same shape as ``X``, each column mean-centered.
    """
    # A8: feed Fortran-order directly so the parallel kernel can slice
    # contiguous columns without an internal .copy().
    S = kalman_smooth_matrix(np.asfortranarray(X, dtype=np.float64), dV=dV, dW=dW)
    S -= S.mean(axis=0, keepdims=True)
    return S
