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
