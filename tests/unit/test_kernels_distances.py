import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from pycopykat.kernels.distances import pdist_euclidean, pdist_pearson, pdist_spearman

rng = np.random.default_rng(0)


def _rand(n=30, p=40):
    return rng.standard_normal((n, p))


def test_euclidean_matches_scipy():
    X = _rand()
    got = pdist_euclidean(X)
    want = pdist(X, "euclidean")
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)


# ---- BLAS-path agreement with scipy.pdist ----
#
# These shapes all exceed the n<100 scipy fallback threshold so the BLAS
# identity path is actually exercised.


@pytest.mark.parametrize("seed,n,p", [(1, 100, 10), (2, 150, 200), (3, 200, 50)])
def test_euclidean_blas_path_matches_scipy_random(seed, n, p):
    X = np.random.default_rng(seed).standard_normal((n, p))
    got = pdist_euclidean(X)
    want = pdist(X, "euclidean")
    # BLAS summation order differs from scipy's naive pair loop; atol=1e-8
    # is ample for downstream Ward cut-point decisions.
    np.testing.assert_allclose(got, want, atol=1e-8, rtol=1e-10)


def test_euclidean_blas_path_all_zeros():
    X = np.zeros((120, 30), dtype=np.float64)
    got = pdist_euclidean(X)
    assert got.shape == (120 * 119 // 2,)
    # Every pair has distance exactly zero (identity + clamp guarantees this).
    assert np.all(got == 0.0)


def test_euclidean_blas_path_duplicated_rows():
    """Identical rows survive the BLAS identity path without NaN/Inf.

    Without the `np.maximum(D2, 0, out=D2)` step, catastrophic cancellation
    in ``||a||^2 + ||b||^2 - 2*a*b^T`` can yield tiny NEGATIVE squared
    distances for near-identical rows — ``np.sqrt`` would then emit NaN.

    The identity formula cannot reproduce scipy's bitwise ``sqrt(0)``
    behaviour for duplicated rows (scipy computes ``sqrt(sum((a-b)**2))``
    which is ``sqrt(0)`` exactly when ``a == b``; the identity evaluates
    three separate sums whose rounding does not cancel). What we DO
    guarantee is: finite, non-negative, near-zero at float32-sqrt precision.
    """
    base = np.random.default_rng(42).standard_normal((1, 50))
    X = np.repeat(base, 110, axis=0)  # 110 copies of the same row
    got = pdist_euclidean(X)
    assert got.shape == (110 * 109 // 2,)
    assert np.all(np.isfinite(got))
    assert np.all(got >= 0.0)
    # sqrt of a clamped ~float64 cancellation residual; 1e-6 is generous.
    assert got.max() < 1e-6


def test_euclidean_blas_path_rank_deficient():
    """Rank-deficient inputs (rows span a low-dim subspace) still agree."""
    rng_ = np.random.default_rng(7)
    basis = rng_.standard_normal((120, 3))            # 3-dim latent
    loadings = rng_.standard_normal((3, 40))
    X = basis @ loadings                              # rank-3 in 40-d
    got = pdist_euclidean(X)
    want = pdist(X, "euclidean")
    np.testing.assert_allclose(got, want, atol=1e-8, rtol=1e-10)


# ---- small-N fallback ----


def test_euclidean_small_n_uses_scipy_fallback():
    """For n<100 we route through scipy; output must be bit-identical."""
    X = np.random.default_rng(123).standard_normal((50, 80))
    got = pdist_euclidean(X)
    want = pdist(np.ascontiguousarray(X, dtype=np.float64), metric="euclidean")
    # Bit-for-bit identical because we delegate directly to scipy.
    np.testing.assert_array_equal(got, want)


# ---- condensed ordering sanity check ----


def test_euclidean_blas_path_ordering_matches_scipy():
    """Verify the triu_indices(k=1) ordering matches scipy's condensed layout.

    Build a small (>= fallback threshold) matrix, compute condensed via
    BLAS and via squareform round-trip, confirm element-wise agreement.
    """
    X = np.random.default_rng(11).standard_normal((105, 20))
    got = pdist_euclidean(X)
    # squareform condenses in the same (i<j, row-major) order as pdist.
    D_full = squareform(pdist(X, "euclidean"))
    iu, ju = np.triu_indices(105, k=1)
    expected = D_full[iu, ju]
    np.testing.assert_allclose(got, expected, atol=1e-8, rtol=1e-10)


def test_pearson_matches_scipy():
    X = _rand()
    got = pdist_pearson(X)  # 1 - r
    want = pdist(X, "correlation")  # scipy's "correlation" is 1 - Pearson
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)


def test_spearman_matches_scipy():
    X = _rand(n=20, p=30)
    got = pdist_spearman(X)
    n = X.shape[0]
    want = np.empty(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(X[i], X[j])
            want[k] = 1.0 - r
            k += 1
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)


def test_shapes():
    X = _rand(n=50, p=10)
    for fn in (pdist_euclidean, pdist_pearson, pdist_spearman):
        out = fn(X)
        assert out.shape == (50 * 49 // 2,)


@pytest.mark.benchmark(group="pearson")
def test_bench_pearson_1k(benchmark):
    X = np.random.default_rng(0).standard_normal((1000, 500))
    benchmark(pdist_pearson, X)
