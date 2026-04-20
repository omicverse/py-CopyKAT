"""Order-1 polynomial (local level) Kalman RTS smoother.

Matches dlm::dlmModPoly(order=1, dV, dW) + dlmSmooth.
Scalar state; closed-form per-step recursions.
"""
# ruff: noqa: N803, N806  # uppercase X/Y/R/C matrix-style names by convention
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _rts_smooth_1d(y: np.ndarray, dV: float, dW: float) -> np.ndarray:
    """
    Forward filter + backward RTS smoother on scalar state with
      x_t = x_{t-1} + w_t, w ~ N(0, dW)
      y_t = x_t + v_t,     v ~ N(0, dV)
    Initial: x_0 ~ N(0, 1e7) (diffuse).
    """
    n = y.shape[0]
    # Forward filter
    m_fwd = np.empty(n + 1)
    C_fwd = np.empty(n + 1)
    a_fwd = np.empty(n + 1)
    R_fwd = np.empty(n + 1)
    m_fwd[0] = 0.0
    C_fwd[0] = 1e7
    for t in range(1, n + 1):
        a = m_fwd[t - 1]
        R = C_fwd[t - 1] + dW
        f = a                      # predicted y
        Q = R + dV
        K = R / Q
        e = y[t - 1] - f
        m_fwd[t] = a + K * e
        C_fwd[t] = R - K * R       # = R * dV / Q
        a_fwd[t] = a
        R_fwd[t] = R
    # Backward smoother (RTS)
    s = np.empty(n + 1)
    s[n] = m_fwd[n]
    for t in range(n - 1, -1, -1):
        if t == 0:
            J = C_fwd[0] / (C_fwd[0] + dW)
            s[0] = m_fwd[0] + J * (s[1] - (m_fwd[0]))
        else:
            J = C_fwd[t] / R_fwd[t + 1]
            s[t] = m_fwd[t] + J * (s[t + 1] - a_fwd[t + 1])
    return s

def kalman_smooth(y: np.ndarray, dV: float = 0.16, dW: float = 0.001) -> np.ndarray:
    """Smooth a 1D series. Returns smoothed series of same length as y
    (R copykat drops s[1] so s[2:] is length n; we replicate that)."""
    s_full = _rts_smooth_1d(np.ascontiguousarray(y, dtype=np.float64), dV, dW)
    return s_full[1:]  # drop the prior x_0 estimate

@njit(cache=True, parallel=True, fastmath=True)
def _smooth_matrix_kernel(Y: np.ndarray, dV: float, dW: float) -> np.ndarray:
    # A8: Y is expected in Fortran order so Y[:, j] is a contiguous view.
    # Avoids the per-column .copy() and the 2×memory temporary.
    g, c = Y.shape
    out = np.empty_like(Y)
    for j in prange(c):
        s_full = _rts_smooth_1d(Y[:, j], dV, dW)
        for i in range(g):
            out[i, j] = s_full[i + 1]
    return out

def kalman_smooth_matrix(Y: np.ndarray, dV: float = 0.16, dW: float = 0.001) -> np.ndarray:
    """Apply smoother column-wise (each cell independent). Parallel over cells.

    The matrix kernel expects Fortran-order input so columns are contiguous
    in memory, which lets the per-column RTS call read straight from ``Y[:, j]``
    without a copy.
    """
    return _smooth_matrix_kernel(np.asfortranarray(Y, dtype=np.float64), dV, dW)
