import numpy as np
import pytest

from pycopykat.kernels.adjust import adjust_threshold


def test_adjust_mutes_near_baseline():
    # g=5 bins, c=2 cells
    X = np.array([[0.0, 0.0, 1.0, 2.0, 3.0],
                  [0.1, 0.1, 1.1, 2.1, 3.1]]).T  # shape (5, 2)
    base = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    sd = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    out = adjust_threshold(X, base, sd, factor=0.25)
    col_means = X.mean(axis=0)
    for j in range(2):
        for i in range(5):
            if abs(X[i, j] - base[i]) < 0.25 * sd[i]:
                assert out[i, j] == pytest.approx(col_means[j])
            else:
                assert out[i, j] == pytest.approx(X[i, j])


def test_adjust_shape_preserved():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 30))
    base = np.zeros(200)
    sd = np.ones(200)
    out = adjust_threshold(X, base, sd, factor=0.25)
    assert out.shape == X.shape
    assert out.dtype == np.float64


def test_adjust_zero_factor_passes_through():
    """factor=0 means no bin is muted — output equals input."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 8))
    base = rng.standard_normal(50)
    sd = np.ones(50)
    out = adjust_threshold(X, base, sd, factor=0.0)
    np.testing.assert_allclose(out, X, rtol=1e-12, atol=0.0)
