"""Per-cell GMM fallback for diploid baseline (R copykat ``baseline.GMM``).

Called when :func:`baseline_norm_cl` flags low confidence. For each cell we
fit a 3-component tied-covariance Gaussian mixture with init means
``(-0.2, 0, 0.2)``. A cell is declared ``diploid`` when at least one component
satisfies ``|μ| ≤ mu_cut`` **and** those components' mixture weights sum to
more than ``nfraq_cut``. The algorithm early-stops as soon as
``max_normal`` diploid cells have been collected.

If fewer than 3 diploids are found the function either returns a caller-supplied
fallback :class:`BaselineResult` (mimicking R's ``return(RE.before)``) or, when
no fallback is provided, an empty result with a low-confidence warning.
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from pycopykat.baseline._shared import BaselineResult, ward_cluster_with_min_size


def baseline_gmm(
    X: NDArray[np.floating],
    cell_names: list[str] | None = None,
    *,
    max_normal: int = 5,
    mu_cut: float = 0.05,
    nfraq_cut: float = 0.99,
    fallback: BaselineResult | None = None,
    seed: int = 1234,
) -> BaselineResult:
    """Locate a small set of diploid cells cell-by-cell via GMM, then average.

    Parameters
    ----------
    X
        ``(n_genes, n_cells)`` smoothed/centered matrix.
    cell_names
        Optional length-``n_cells`` identifiers.
    max_normal
        Early-stop once this many diploid cells have been collected.
    mu_cut
        A component counts as "near zero" if ``|mean| ≤ mu_cut``.
    nfraq_cut
        A cell is diploid when near-zero components' total weight ``> nfraq_cut``.
    fallback
        Returned verbatim when fewer than 3 diploid cells are detected — this
        mirrors R copykat's ``return(RE.before)`` behaviour. Pass ``None`` to
        receive an empty, ``"unclassified.prediction"`` result instead.
    seed
        Random seed for each per-cell GMM.
    """
    n_genes, n_cells = X.shape
    if cell_names is None:
        cell_names = [f"c{i}" for i in range(n_cells)]
    if len(cell_names) != n_cells:
        raise ValueError(f"cell_names length {len(cell_names)} != {n_cells} cells")

    diploid_names: list[str] = []

    for m in range(n_cells):
        sam = X[:, m].astype(np.float64)
        sg = max(0.05, 0.5 * float(np.std(sam, ddof=1)))
        gmm = GaussianMixture(
            n_components=3,
            means_init=np.array([[-0.2], [0.0], [0.2]]),
            covariance_type="tied",
            max_iter=500,
            random_state=seed,
            reg_covar=1e-8,
        )
        try:
            gmm.precisions_init = np.array([[1.0 / (sg ** 2)]])
        except Exception:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gmm.fit(sam.reshape(-1, 1))

        mus = gmm.means_.ravel()
        lambdas = gmm.weights_.ravel()
        near_zero = np.abs(mus) <= mu_cut

        if not near_zero.any():
            pred = "aneuploid"
        else:
            frq = float(lambdas[near_zero].sum())
            pred = "diploid" if frq > nfraq_cut else "aneuploid"

        if pred == "diploid":
            diploid_names.append(cell_names[m])
            if len(diploid_names) >= max_normal:
                break

    # Ward k=6 (no backoff in R) for the labels field; clamp to n_cells.
    labels, _Z = ward_cluster_with_min_size(
        X.T, initial_k=min(6, max(2, n_cells - 1)),
        min_cells=1, distance="euclidean", k_floor=2,
    )

    if len(diploid_names) > 2:
        diploid_set = set(diploid_names)
        mask = np.array([cn in diploid_set for cn in cell_names])
        basel = X[:, mask].mean(axis=1).astype(np.float64)
        return BaselineResult(
            basel=basel, preN=diploid_names, warning="", labels=labels,
        )

    if fallback is not None:
        return fallback

    return BaselineResult(
        basel=np.zeros(n_genes, dtype=np.float64),
        preN=[],
        warning="unclassified.prediction",
        labels=labels,
    )
