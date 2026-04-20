"""Tests for pycopykat.preprocess.normalize — VST + per-cell centering."""
import numpy as np
from scipy import sparse

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


# ─── S2: sparse-path parity ─────────────────────────────────────────────────


def test_vst_preserves_zeros_invariant():
    # The algebraic invariant that makes the sparse path safe.
    assert np.log(np.sqrt(0.0) + np.sqrt(0.0 + 1.0)) == 0.0


def test_vst_center_sparse_matches_dense_large_random():
    """Gate 4: sparse output close to dense on a real-sized sparse random mat.

    Shape ~5000×500 @ 10% density; allclose at 1e-12 because the two paths
    summation orders differ (dense iterates all rows; sparse iterates nnz).
    """
    rng = np.random.default_rng(2026)
    n_genes, n_cells = 5000, 500
    density = 0.10
    mask = rng.random((n_genes, n_cells)) < density
    vals = rng.poisson(3, size=(n_genes, n_cells)).astype(np.int32)
    dense = np.where(mask, vals, 0).astype(np.float64)
    X_csr = sparse.csr_matrix(dense)
    X_csc = sparse.csc_matrix(dense)

    out_dense = vst_center(dense)
    out_csr = vst_center(X_csr)
    out_csc = vst_center(X_csc)

    assert out_csr.shape == out_dense.shape
    assert out_csc.shape == out_dense.shape
    assert out_csr.dtype == np.float64
    assert out_csc.dtype == np.float64
    np.testing.assert_allclose(out_csr, out_dense, atol=1e-12)
    np.testing.assert_allclose(out_csc, out_dense, atol=1e-12)


def test_vst_center_sparse_does_not_mutate_caller():
    """Caller's sparse matrix data must be untouched after vst_center."""
    rng = np.random.default_rng(3)
    dense = (rng.random((50, 20)) < 0.2).astype(np.int32) * rng.integers(
        1, 10, size=(50, 20)
    ).astype(np.int32)
    X = sparse.csc_matrix(dense)
    data_before = X.data.copy()
    indices_before = X.indices.copy()
    indptr_before = X.indptr.copy()

    _ = vst_center(X)

    np.testing.assert_array_equal(X.data, data_before)
    np.testing.assert_array_equal(X.indices, indices_before)
    np.testing.assert_array_equal(X.indptr, indptr_before)


def test_vst_center_sparse_columns_mean_zero():
    rng = np.random.default_rng(11)
    dense = (rng.random((200, 30)) < 0.15).astype(np.int32) * rng.integers(
        1, 20, size=(200, 30)
    ).astype(np.int32)
    out = vst_center(sparse.csc_matrix(dense.astype(np.float64)))
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)
