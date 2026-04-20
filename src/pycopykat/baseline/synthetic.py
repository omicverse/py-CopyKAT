"""Cell-line (synthetic normal) baseline — R copykat ``baseline.synthetic``.

When no real diploid cells exist (e.g. pure cancer cell-line experiments),
copykat fabricates a per-cluster synthetic "normal" profile: for every gene,
sample a single value from ``N(0, σ_gene)`` where ``σ_gene`` is that gene's
standard deviation across cells in the cluster. Each cell then gets its
synthetic-subtracted relative expression.

The output schema differs from :func:`baseline_norm_cl` and :func:`baseline_gmm`:
there is no single ``basel`` vector — each cluster contributes one synthetic
profile (stored column-wise in ``syn_normal``), and every cell receives a
relative-expression column in ``expr_relat``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pycopykat.baseline._shared import ward_cluster_with_min_size


@dataclass(slots=True)
class SyntheticBaselineResult:
    """Output of :func:`baseline_synthetic`.

    Attributes
    ----------
    expr_relat
        ``(n_genes, n_cells)`` relative expression with a per-cluster synthetic
        normal subtracted. Columns are ordered cluster-by-cluster (cluster 1
        first, then 2, …) — see ``cell_order`` for the permutation.
    syn_normal
        ``(n_genes, n_clusters)`` synthetic normal profiles; one column per
        cluster in increasing label order.
    labels
        Cluster labels in the **original** input column order (1-based).
    cell_order
        Names of the cells making up ``expr_relat`` columns in emitted order.
    """

    expr_relat: NDArray[np.float64]
    syn_normal: NDArray[np.float64]
    labels: NDArray[np.int_]
    cell_order: list[str]


def baseline_synthetic(
    X: NDArray[np.floating],
    cell_names: list[str] | None = None,
    *,
    min_cells: int = 10,
    seed: int = 123,
) -> SyntheticBaselineResult:
    """Generate per-cluster synthetic normal baselines.

    Parameters
    ----------
    X
        ``(n_genes, n_cells)`` smoothed/centered matrix.
    cell_names
        Optional length-``n_cells`` identifiers; defaults to ``c0 … c{n-1}``.
    min_cells
        Minimum cells per cluster after backoff loop (R default 10).
    seed
        RNG seed for the synthetic samples. Matches R's behaviour of calling
        ``set.seed`` once per cluster so every cluster starts from the same
        N(0,1) stream.
    """
    n_genes, n_cells = X.shape
    if cell_names is None:
        cell_names = [f"c{i}" for i in range(n_cells)]
    if len(cell_names) != n_cells:
        raise ValueError(f"cell_names length {len(cell_names)} != {n_cells} cells")

    labels, _Z, _d = ward_cluster_with_min_size(
        X.T, initial_k=6, min_cells=min_cells + 1,
        distance="euclidean", k_floor=2,
    )
    unique_labs = np.unique(labels)

    expr_blocks: list[NDArray[np.float64]] = []
    syn_blocks: list[NDArray[np.float64]] = []
    emitted: list[str] = []

    for lab in unique_labs:
        mask = labels == lab
        cluster_cells = X[:, mask].astype(np.float64)
        sd1 = cluster_cells.std(axis=1, ddof=1)
        # Reseed every cluster to mirror R's per-iteration set.seed(123)
        rng = np.random.default_rng(seed)
        syn_norm = rng.normal(loc=0.0, scale=sd1, size=sd1.shape)
        relat = cluster_cells - syn_norm[:, None]
        expr_blocks.append(relat)
        syn_blocks.append(syn_norm)
        emitted.extend([n for n, m in zip(cell_names, mask) if m])

    expr_relat = np.hstack(expr_blocks).astype(np.float64)
    syn_normal = np.column_stack(syn_blocks).astype(np.float64)

    return SyntheticBaselineResult(
        expr_relat=expr_relat,
        syn_normal=syn_normal,
        labels=labels,
        cell_order=emitted,
    )
