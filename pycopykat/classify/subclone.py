"""Simplified WGCNA dynamicTreeCut hybrid method (Langfelder et al. 2008).

This is a V1 port that covers the common use-case of finding 2-6 well-separated
subclones in a Ward hierarchical clustering. It is NOT a bit-exact port of the
R ``dynamicTreeCut`` package — see Task 6.2 Step 6 of the implementation plan.

Algorithm
---------
Walk the linkage tree top-down from the root. At each internal node:

* If both child subtrees are at least ``min_cluster_size`` AND the merge is
  above the ``deep_split``-controlled height threshold, descend recursively.
* Otherwise accept the current subtree as one cluster (if it meets the size
  threshold).

Leaves in subtrees that never reach ``min_cluster_size`` receive label ``0``
("unassigned", analogous to R's cluster 0).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _descendants_cache(Z: NDArray[np.floating], n: int) -> dict[int, list[int]]:
    """Precompute leaf descendants for every internal node iteratively."""
    cache: dict[int, list[int]] = {i: [i] for i in range(n)}
    for k in range(Z.shape[0]):
        left = int(Z[k, 0])
        right = int(Z[k, 1])
        cache[n + k] = cache[left] + cache[right]
    return cache


def dynamic_tree_cut(
    Z: NDArray[np.floating],
    *,
    min_cluster_size: int = 10,
    deep_split: int = 2,
) -> NDArray[np.int_]:
    """Label each leaf with a cluster id; ``0`` = unassigned.

    Parameters
    ----------
    Z
        scipy linkage matrix (``(n-1, 4)``).
    min_cluster_size
        Minimum size a subtree must reach to become its own cluster.
    deep_split
        ``0..4``; larger values permit more splits at shallower heights.

    Returns
    -------
    Length-``n`` ``int`` array of cluster labels (1-based; 0 for unassigned).
    """
    n = Z.shape[0] + 1
    heights = Z[:, 2]
    h_max = float(heights.max())
    h_min = float(heights.min())
    cut_h = h_min + (h_max - h_min) * (1.0 - 0.15 * (deep_split + 1))

    descendants = _descendants_cache(Z, n)
    labels = np.zeros(n, dtype=np.int_)

    # Iterative traversal using an explicit stack to avoid recursion depth
    # issues on large trees (~30k cells).
    next_label = 1
    root = n + Z.shape[0] - 1
    stack: list[int] = [root]
    while stack:
        node = stack.pop()
        leaves = descendants[node]
        if len(leaves) < min_cluster_size:
            continue
        if node < n:
            # single leaf — can never form a cluster on its own
            continue
        merge_h = float(Z[node - n, 2])
        left = int(Z[node - n, 0])
        right = int(Z[node - n, 1])
        l_size = len(descendants[left])
        r_size = len(descendants[right])
        if merge_h > cut_h and l_size >= min_cluster_size and r_size >= min_cluster_size:
            stack.append(left)
            stack.append(right)
            continue
        # Accept current subtree as one cluster
        for leaf in leaves:
            if labels[leaf] == 0:
                labels[leaf] = next_label
        next_label += 1

    return labels
