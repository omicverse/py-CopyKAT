"""Pre-final-classification baseline adjustment (R copykat.R ~L371-391).

Between the first Ward k=2 cut (:func:`predict_ploidy`) and the final one,
R copykat:

1. Subtracts the diploid cluster's per-bin mean from every cell.
2. Centers columns at zero.
3. Computes the diploid cluster's per-bin standard deviation ``cf.h``.
4. For every cell, mutes entries whose deviation from the diploid mean is
   within ``factor * cf.h`` (default 0.25): those entries are replaced by
   the cell's column mean. This is the per-cell threshold kernel provided
   by :mod:`pycopykat.kernels.adjust`.
5. Re-centers columns at zero.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pycopykat.kernels.adjust import adjust_threshold


def baseline_adjust(
    cna: NDArray[np.floating],
    diploid_mask: NDArray[np.bool_],
    *,
    factor: float = 0.25,
) -> NDArray[np.float64]:
    """Diploid-anchored CNA adjustment before the final clustering pass.

    Parameters
    ----------
    cna
        ``(n_bins, n_cells)`` CNA matrix.
    diploid_mask
        Length-``n_cells`` boolean mask of cells previously called diploid.
    factor
        Threshold multiplier on ``cf.h`` (default 0.25, matches R).
    """
    diploid_mask = np.asarray(diploid_mask, dtype=bool)
    if int(diploid_mask.sum()) < 2:
        raise ValueError(
            "baseline_adjust needs at least 2 diploid cells for per-bin sd"
        )

    cna = np.asarray(cna, dtype=np.float64)

    # 1. subtract diploid per-bin mean
    dip_mean = cna[:, diploid_mask].mean(axis=1, keepdims=True)
    rel = cna - dip_mean

    # 2. center columns at zero
    rel -= rel.mean(axis=0, keepdims=True)

    # 3. per-bin sd + mean on the diploid subset
    dip_rel = rel[:, diploid_mask]
    sd = dip_rel.std(axis=1, ddof=1)
    base = dip_rel.mean(axis=1)

    # 4. threshold-mute small deviations (Numba kernel)
    out = adjust_threshold(rel, base=base, sd=sd, factor=factor)

    # 5. re-center columns at zero
    out = np.asarray(out, dtype=np.float64)
    out -= out.mean(axis=0, keepdims=True)
    return out
