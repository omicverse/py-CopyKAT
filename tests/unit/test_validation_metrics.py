"""Tests for pycopykat.validation.metrics — R vs Python comparison metrics."""
import numpy as np
import pandas as pd

from pycopykat.validation.metrics import compare_cna, compare_predictions


def test_compare_predictions_matches_and_penalises_mismatch():
    pred_r = pd.DataFrame({
        "cell": [f"c{i}" for i in range(20)],
        "copykat.pred": ["diploid"] * 10 + ["aneuploid"] * 10,
    })
    pred_py = pd.DataFrame({
        "cell": [f"c{i}" for i in range(20)],
        "copykat.pred": ["diploid"] * 9 + ["aneuploid"] * 11,
    })
    m = compare_predictions(pred_r, pred_py)
    assert m["n_shared"] == 20
    # one of 20 cells flipped → ARI around 0.8
    assert m["ari"] >= 0.75
    assert 0.0 <= m["kappa"] <= 1.0
    assert 0.0 <= m["fmi"] <= 1.0


def test_compare_predictions_perfect():
    pred = pd.DataFrame({
        "cell": [f"c{i}" for i in range(6)],
        "copykat.pred": ["diploid"] * 3 + ["aneuploid"] * 3,
    })
    m = compare_predictions(pred, pred.copy())
    assert m["ari"] == 1.0
    assert m["kappa"] == 1.0


def test_compare_predictions_cell_subset():
    pred_r = pd.DataFrame({
        "cell": [f"c{i}" for i in range(5)],
        "copykat.pred": ["diploid"] * 5,
    })
    pred_py = pd.DataFrame({
        "cell": [f"c{i}" for i in range(3, 8)],
        "copykat.pred": ["diploid"] * 5,
    })
    m = compare_predictions(pred_r, pred_py)
    assert m["n_shared"] == 2


def test_compare_cna_spearman():
    rng = np.random.default_rng(0)
    cna_r = pd.DataFrame(
        rng.standard_normal((100, 10)),
        columns=[f"c{i}" for i in range(10)],
    )
    cna_py = cna_r + rng.standard_normal(cna_r.shape) * 0.05
    m = compare_cna(cna_r, cna_py, method="spearman")
    assert m["median_r"] > 0.9
    assert m["n_shared"] == 10


def test_compare_cna_no_overlap():
    cna_r = pd.DataFrame(np.zeros((5, 2)), columns=["a", "b"])
    cna_py = pd.DataFrame(np.zeros((5, 2)), columns=["x", "y"])
    m = compare_cna(cna_r, cna_py)
    assert m["n_shared"] == 0
    assert np.isnan(m["median_r"])
