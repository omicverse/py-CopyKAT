"""Diploid vs aneuploid per-cell prediction (R copykat.R ~L345-399).

After CNA estimation, R copykat cuts a Ward hierarchical clustering of the
cells' binned CNA matrix at ``k = 2`` and declares the cluster with the
higher fraction of pre-identified normal cells (``preN``) to be diploid.
We reproduce that with scipy's Ward linkage and a euclidean/pearson/spearman
distance choice consistent with the rest of the package.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage

from pycopykat.kernels.distances import (
    pdist_euclidean,
    pdist_pearson,
    pdist_spearman,
)

_DIST_FN: dict[str, Callable[[NDArray[np.floating]], NDArray[np.float64]]] = {
    "euclidean": pdist_euclidean,
    "pearson": pdist_pearson,
    "spearman": pdist_spearman,
}


def predict_ploidy(
    cna: NDArray[np.floating],
    cells: list[str],
    *,
    preN: list[str],
    distance: str = "euclidean",
) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    """Assign each cell to ``"diploid"`` or ``"aneuploid"``.

    Parameters
    ----------
    cna
        ``(n_bins, n_cells)`` adjusted CNA matrix.
    cells
        Length-``n_cells`` cell identifiers, in column order.
    preN
        Names of cells that auto/GMM/synthetic baseline flagged as diploid.
    distance
        ``"euclidean"``, ``"pearson"`` or ``"spearman"``.

    Returns
    -------
    (pred, Z)
        ``pred`` is a DataFrame with columns ``cell`` and ``copykat.pred``
        in the original cell order. ``Z`` is the scipy Ward linkage matrix
        computed on ``cna.T`` with the requested ``distance``.

        The returned linkage is valid for reuse downstream only when both
        the input matrix AND the distance function are unchanged.
    """
    if len(cells) != cna.shape[1]:
        raise ValueError(f"cells length {len(cells)} != cna cols {cna.shape[1]}")
    if distance not in _DIST_FN:
        raise ValueError(f"distance must be one of {list(_DIST_FN)}, got {distance!r}")

    d = _DIST_FN[distance](np.asarray(cna.T, dtype=np.float64))
    Z = linkage(d, method="ward")
    labels = fcluster(Z, t=2, criterion="maxclust")

    preN_set = set(preN)
    frac_normal: dict[int, float] = {}
    for lab in np.unique(labels):
        members = [cells[i] for i, l in enumerate(labels) if l == lab]
        n = len(members)
        frac_normal[int(lab)] = (
            sum(1 for m in members if m in preN_set) / max(1, n)
        )

    if len(frac_normal) == 1:
        # Only one cluster — degenerate; call every cell diploid since preN ⊂ it
        pred = ["diploid"] * len(cells)
    else:
        diploid_label = max(frac_normal, key=lambda k: frac_normal[k])
        pred = [
            "diploid" if int(l) == diploid_label else "aneuploid" for l in labels
        ]
    return pd.DataFrame({"cell": cells, "copykat.pred": pred}), Z
