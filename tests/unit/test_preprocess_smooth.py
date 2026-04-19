"""Tests for pycopykat.preprocess.smooth — per-cell Kalman + post-center."""
import numpy as np

from pycopykat.preprocess.smooth import smooth_cells


def test_output_shape_and_centered():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 8))
    S = smooth_cells(X)
    assert S.shape == X.shape
    np.testing.assert_allclose(S.mean(axis=0), 0.0, atol=1e-10)


def test_smoothing_reduces_variance_on_noisy_constant():
    # A smoother applied to gaussian noise around 0 should shrink pointwise var
    rng = np.random.default_rng(1)
    X = rng.standard_normal((1000, 4))
    S = smooth_cells(X)
    assert (S.var(axis=0) < X.var(axis=0)).all()


def test_accepts_float_dtype():
    X = np.ones((100, 2), dtype=np.float64)
    S = smooth_cells(X)
    # All-ones input smoothed then centered → all zeros (up to Kalman fp error)
    np.testing.assert_allclose(S, 0.0, atol=1e-8)
