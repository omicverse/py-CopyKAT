"""Tests for pycopykat.preprocess.normalize — VST + per-cell centering."""
import numpy as np

from pycopykat.preprocess.normalize import vst_center


def test_vst_formula_on_small_case():
    x = np.array([[0.0, 1.0], [4.0, 9.0]])
    want_raw = np.log(np.sqrt(x) + np.sqrt(x + 1))
    want = want_raw - want_raw.mean(axis=0)
    got = vst_center(x)
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_columns_are_mean_zero():
    rng = np.random.default_rng(0)
    x = rng.poisson(5, size=(100, 10)).astype(float)
    out = vst_center(x)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)


def test_preserves_shape_and_dtype():
    x = np.zeros((50, 3), dtype=np.int64)
    out = vst_center(x)
    assert out.shape == (50, 3)
    assert out.dtype == np.float64


def test_zero_input_reduces_to_log1_mean_zero():
    # VST of 0 = log(0 + 1) = 0; centering of all-zeros is all-zeros
    out = vst_center(np.zeros((5, 2)))
    np.testing.assert_allclose(out, 0.0, atol=1e-12)
