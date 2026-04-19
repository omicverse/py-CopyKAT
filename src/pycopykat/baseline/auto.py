"""Auto-mode diploid baseline estimation (R copykat ``baseline.norm.cl``).

Algorithm (verbatim against R copykat baseline.norm.cl.R):

1. Ward hierarchical clustering of cells (euclidean distance) starting at
   ``k = 6``; decrement until every cluster has at least ``min_cells`` members,
   stopping at ``k_floor = 2``.
2. For each cluster, take the **per-gene median** across the cluster's cells
   (NOT the raveled cell × gene values — this is the 1-D signal R passes to
   mixtools::normalmixEM).
3. Fit a 3-component Gaussian mixture with **tied covariance** (matches R's
   ``arbvar=FALSE``) and initial means ``[-0.2, 0, 0.2]``. Record ``sigma`` as
   the single shared standard deviation.
4. Silhouette quality ``wn = mean silhouette at k=2`` — not the K used above.
5. F-test on the sigma ratio: ``PDt = 1 - F.cdf(σ_max² / σ_min², n_genes, n_genes)``.
6. ``warning = "unclassified.prediction"`` if ``wn ≤ 0.15``, or any cluster
   still has ``≤ min_cells`` after the backoff, or ``PDt > 0.05``; else ``""``.
7. Baseline = per-gene median over cells belonging to any cluster whose sigma
   equals the minimum (handles ties explicitly like R ``ct %in% which(SDM==min(SDM))``).
"""
from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.stats import f as f_dist
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture

from pycopykat.baseline._shared import BaselineResult, ward_cluster_with_min_size
from pycopykat.kernels.distances import pdist_euclidean


def baseline_norm_cl(
    X: NDArray[np.floating],
    cell_names: list[str] | None = None,
    *,
    min_cells: int = 5,
    seed: int = 1234,
    **_unused: object,
) -> BaselineResult:
    """Auto-estimate a diploid baseline via Ward + tied-covariance GMM.

    Parameters
    ----------
    X
        ``(n_genes, n_cells)`` VST-centered, Kalman-smoothed matrix.
    cell_names
        Optional length-``n_cells`` identifiers. Defaults to ``c0 … c{n-1}``.
    min_cells
        Minimum cells per cluster after the backoff loop.
    seed
        Random seed for the ``GaussianMixture`` fits.
    """
    n_genes, n_cells = X.shape
    if cell_names is None:
        cell_names = [f"c{i}" for i in range(n_cells)]
    if len(cell_names) != n_cells:
        raise ValueError(f"cell_names length {len(cell_names)} != {n_cells} cells")

    # ---- step 1: Ward cluster in cell-space with k floor at 2 ----
    # R uses `table(ct) > min.cells` (strictly greater) as the backoff test, so
    # pass min_cells + 1 to the helper (which uses `>=`) to match R semantics.
    labels, Z = ward_cluster_with_min_size(
        X.T, initial_k=6, min_cells=min_cells + 1,
        distance="euclidean", k_floor=2,
    )
    unique_labs = np.unique(labels)
    all_meet_min = all(int((labels == lab).sum()) > min_cells for lab in unique_labs)

    # ---- steps 2-3: per-cluster per-gene median -> tied-cov 3-GMM -> sigma ----
    SDM: list[float] = []
    for lab in unique_labs:
        cluster_cells_mask = labels == lab
        data_c = np.median(X[:, cluster_cells_mask], axis=1).astype(np.float64)
        sx = max(0.05, 0.5 * float(np.std(data_c, ddof=1)))
        gmm = GaussianMixture(
            n_components=3,
            means_init=np.array([[-0.2], [0.0], [0.2]]),
            covariance_type="tied",
            max_iter=5000,
            random_state=seed,
            reg_covar=1e-8,
        )
        # initialize the tied covariance to sx**2 via precisions_init for stability
        try:
            gmm.precisions_init = np.array([[1.0 / (sx ** 2)]])
        except Exception:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gmm.fit(data_c.reshape(-1, 1))
        # tied covariance → single (1,1) matrix
        sigma = float(np.sqrt(np.asarray(gmm.covariances_).ravel()[0]))
        SDM.append(sigma)

    SDM_arr = np.asarray(SDM, dtype=np.float64)

    # ---- step 4: silhouette at k=2 on the same Ward tree ----
    d = pdist_euclidean(X.T.astype(np.float64))
    D = squareform(d)
    labels_k2 = fcluster(Z, t=2, criterion="maxclust")
    if np.unique(labels_k2).size >= 2:
        sil_k2 = float(np.mean(silhouette_samples(D, labels_k2, metric="precomputed")))
    else:
        sil_k2 = 1.0  # degenerate: only one cluster possible

    # ---- step 5: F-test on sigma ratio ----
    if SDM_arr.min() > 0:
        PDt = float(1.0 - f_dist.cdf(
            (SDM_arr.max() ** 2) / (SDM_arr.min() ** 2), n_genes, n_genes
        ))
    else:
        PDt = 0.0

    # ---- step 6: warning flag ----
    warning = ""
    if sil_k2 <= 0.15 or (not all_meet_min) or PDt > 0.05:
        warning = "unclassified.prediction"

    # ---- step 7: baseline from minimum-sigma cluster(s) ----
    min_sigma = SDM_arr.min()
    baseline_lab_idx = np.where(SDM_arr == min_sigma)[0]
    baseline_cluster_labels = unique_labs[baseline_lab_idx]
    baseline_mask = np.isin(labels, baseline_cluster_labels)
    basel = np.median(X[:, baseline_mask], axis=1).astype(np.float64)
    preN = [cn for cn, keep in zip(cell_names, baseline_mask) if keep]

    return BaselineResult(basel=basel, preN=preN, warning=warning, labels=labels)
