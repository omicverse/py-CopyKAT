"""Tests for pycopykat.baseline.auto — auto diploid baseline estimation."""
import numpy as np

from pycopykat.baseline.auto import baseline_norm_cl


def test_identifies_low_variance_cluster_as_baseline():
    rng = np.random.default_rng(0)
    g = 500
    # 30 diploid-like cells (low noise, centered at 0)
    # + 30 aneuploid-like cells (high noise, shifted)
    dip = rng.normal(0,   0.05, size=(g, 30))
    ane = rng.normal(0.3, 0.3,  size=(g, 30))
    X = np.hstack([dip, ane])
    cell_names = [f"d{i}" for i in range(30)] + [f"a{i}" for i in range(30)]

    result = baseline_norm_cl(X, cell_names=cell_names, min_cells=5, seed=1234)

    # The diploid cells should dominate result.preN (they're the low-sigma group)
    n_d_in = sum(1 for n in result.preN if n.startswith("d"))
    n_a_in = sum(1 for n in result.preN if n.startswith("a"))
    assert n_d_in >= n_a_in, (
        f"expected diploid cells to dominate baseline, got d={n_d_in} a={n_a_in}"
    )
    # basel should be roughly gene-wise median of selected cells
    assert result.basel.shape == (g,)
    # warning is either "" (ok) or "unclassified.prediction"
    assert result.warning in ("", "unclassified.prediction")
    # labels cover all 60 cells
    assert result.labels.shape == (60,)


def test_basel_shape_and_preN_subset():
    rng = np.random.default_rng(42)
    g = 200
    X = rng.normal(0, 0.1, size=(g, 40))
    names = [f"c{i}" for i in range(40)]
    res = baseline_norm_cl(X, cell_names=names, min_cells=5, seed=7)
    assert res.basel.shape == (g,)
    assert set(res.preN).issubset(set(names))
    assert len(res.preN) >= 5  # at least one small cluster


def test_default_cell_names():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 0.1, size=(80, 30))
    res = baseline_norm_cl(X, min_cells=5, seed=0)
    # Should auto-generate c0 … c29
    assert all(n.startswith("c") for n in res.preN)
