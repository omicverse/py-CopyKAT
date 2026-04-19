"""Shared baseline utilities.

* :func:`ward_cluster_with_min_size` — Ward hierarchical clustering with a
  minimum cells-per-cluster constraint (R copykat backoff loop).
* :class:`BaselineResult` — container returned by every baseline estimator
  (auto / gmm / synthetic).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage

from pycopykat.kernels.distances import (
    pdist_euclidean,
    pdist_pearson,
    pdist_spearman,
)


@dataclass(slots=True)
class BaselineResult:
    """Outputs shared by all baseline estimators.

    Attributes
    ----------
    basel
        Per-gene baseline vector.
    preN
        Names of cells contributing to ``basel``.
    warning
        ``""`` on success, ``"unclassified.prediction"`` on low confidence.
    labels
        Cluster labels assigned to every input cell (1-based).
    """

    basel: NDArray[np.float64]
    preN: list[str]
    warning: str
    labels: NDArray[np.int_]

_DIST_FN: dict[str, Callable[[NDArray[np.floating]], NDArray[np.float64]]] = {
    "euclidean": pdist_euclidean,
    "pearson": pdist_pearson,
    "spearman": pdist_spearman,
}


def ward_cluster_with_min_size(
    X: NDArray[np.floating],
    *,
    initial_k: int = 6,
    min_cells: int = 10,
    distance: str = "euclidean",
    k_floor: int = 1,
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Ward-link hierarchical clustering honouring a minimum cluster size.

    Parameters
    ----------
    X
        ``(n_points, n_features)`` matrix to cluster.
    initial_k
        Starting number of cuts. Decremented by one whenever the smallest
        resulting cluster has fewer than ``min_cells`` members.
    min_cells
        Minimum members required per cluster.
    distance
        One of ``"euclidean"``, ``"pearson"``, ``"spearman"``.
    k_floor
        Minimum k the loop will step down to before giving up. ``k_floor=1``
        returns a single cluster label when the constraint can't be honoured;
        ``k_floor=2`` mirrors R copykat's ``baseline.norm.cl`` (``if km==2 break``).

    Returns
    -------
    (labels, Z)
        ``labels`` is a length-``n_points`` 1-based integer array.
        ``Z`` is the scipy linkage matrix (``n-1`` rows).
    """
    if distance not in _DIST_FN:
        raise ValueError(f"distance must be one of {list(_DIST_FN)}, got {distance!r}")
    if k_floor < 1:
        raise ValueError(f"k_floor must be ≥ 1, got {k_floor}")

    d = _DIST_FN[distance](np.asarray(X, dtype=np.float64))
    Z = linkage(d, method="ward")

    k = int(initial_k)
    while k > k_floor:
        labels = fcluster(Z, t=k, criterion="maxclust")
        counts = np.bincount(labels)[1:]
        if counts.size > 0 and counts.min() >= min_cells:
            return labels.astype(np.int_), Z
        k -= 1

    # Hit the floor — return whatever that cut yields.
    if k_floor == 1:
        return np.ones(X.shape[0], dtype=np.int_), Z
    labels = fcluster(Z, t=k_floor, criterion="maxclust")
    return labels.astype(np.int_), Z
