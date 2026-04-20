"""Tests for pycopykat.baseline._shared — Ward cluster with min cluster size."""
import numpy as np
import pytest

from pycopykat.baseline._shared import ward_cluster_with_min_size


def test_returns_labels_honoring_min_size():
    rng = np.random.default_rng(0)
    # 4 well-separated blobs of 25 cells each
    X = np.vstack([
        rng.normal(loc=0,  scale=0.1, size=(25, 10)),
        rng.normal(loc=5,  scale=0.1, size=(25, 10)),
        rng.normal(loc=10, scale=0.1, size=(25, 10)),
        rng.normal(loc=15, scale=0.1, size=(25, 10)),
    ])
    labels, Z, d = ward_cluster_with_min_size(
        X, initial_k=6, min_cells=10, distance="euclidean"
    )
    # scipy linkage returns n-1 rows
    assert Z.shape[0] == X.shape[0] - 1
    # condensed distance vector length n*(n-1)/2
    n = X.shape[0]
    assert d.shape == (n * (n - 1) // 2,)
    for lab in np.unique(labels):
        assert (labels == lab).sum() >= 10


def test_backs_off_when_initial_k_exceeds_feasible():
    # 20 points in 2 tight blobs — can't honour k=6 with min_cells=8
    rng = np.random.default_rng(1)
    X = np.vstack([
        rng.normal(0, 0.05, size=(10, 4)),
        rng.normal(5, 0.05, size=(10, 4)),
    ])
    labels, _Z, _d = ward_cluster_with_min_size(
        X, initial_k=6, min_cells=8, distance="euclidean"
    )
    assert len(np.unique(labels)) == 2


def test_degenerate_returns_single_label():
    # 5 points, min_cells=10 → impossible at any k ≥ 2
    X = np.arange(20, dtype=float).reshape(5, 4)
    labels, _Z, _d = ward_cluster_with_min_size(
        X, initial_k=3, min_cells=10, distance="euclidean"
    )
    assert np.unique(labels).size == 1


@pytest.mark.parametrize("distance", ["euclidean", "pearson", "spearman"])
def test_accepts_all_distance_modes(distance):
    rng = np.random.default_rng(2)
    X = np.vstack([
        rng.normal(0, 0.1, size=(15, 20)),
        rng.normal(3, 0.1, size=(15, 20)),
    ])
    labels, _Z, _d = ward_cluster_with_min_size(
        X, initial_k=4, min_cells=5, distance=distance
    )
    assert labels.shape == (30,)
    for lab in np.unique(labels):
        assert (labels == lab).sum() >= 5
