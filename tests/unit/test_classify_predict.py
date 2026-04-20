"""Tests for pycopykat.classify.predict — diploid/aneuploid per-cell call."""
import numpy as np

from pycopykat.classify.predict import predict_ploidy


def test_labels_known_normal_majority_cluster_as_diploid():
    rng = np.random.default_rng(0)
    # 20 cells, 50 bins — two blobs: 10 diploid-like + 10 aneuploid-like
    cna = np.hstack([
        rng.normal(0,   0.05, size=(50, 10)),
        rng.normal(0.3, 0.2,  size=(50, 10)),
    ])
    cells = [f"d{i}" for i in range(10)] + [f"a{i}" for i in range(10)]
    preN = [f"d{i}" for i in range(10)]

    pred, Z = predict_ploidy(cna, cells, preN=preN, distance="euclidean")
    assert list(pred.columns) == ["cell", "copykat.pred"]
    assert Z.shape == (cna.shape[1] - 1, 4)
    dip = set(pred.loc[pred["copykat.pred"] == "diploid", "cell"])
    assert len(dip & set(preN)) >= 8


def test_preserves_cell_order():
    rng = np.random.default_rng(1)
    cna = rng.normal(0, 0.1, size=(30, 12))
    cells = [f"c{i}" for i in range(12)]
    preN = cells[:3]
    pred, _Z = predict_ploidy(cna, cells, preN=preN, distance="euclidean")
    assert list(pred["cell"]) == cells


def test_supports_correlation_distances():
    rng = np.random.default_rng(2)
    cna = np.hstack([
        rng.normal(0,   0.05, size=(40, 8)),
        rng.normal(0.5, 0.2,  size=(40, 8)),
    ])
    cells = [f"d{i}" for i in range(8)] + [f"a{i}" for i in range(8)]
    preN = cells[:8]
    for dist in ("euclidean", "pearson", "spearman"):
        pred, _Z = predict_ploidy(cna, cells, preN=preN, distance=dist)
        labels = pred["copykat.pred"].unique().tolist()
        assert set(labels).issubset({"diploid", "aneuploid"})
