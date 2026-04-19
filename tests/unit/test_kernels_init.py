"""Sanity-check that all kernel helpers can be imported from the package root."""


def test_public_api_importable():
    from pycopykat.kernels import (
        adjust_threshold,
        kalman_smooth,
        kalman_smooth_matrix,
        pdist_euclidean,
        pdist_pearson,
        pdist_spearman,
        pg_posterior_mean,
        pg_posterior_samples,
    )

    # __all__ is the single source of truth
    import pycopykat.kernels as pk

    assert set(pk.__all__) == {
        "adjust_threshold",
        "kalman_smooth",
        "kalman_smooth_matrix",
        "pdist_euclidean",
        "pdist_pearson",
        "pdist_spearman",
        "pg_posterior_mean",
        "pg_posterior_samples",
    }
    for name in pk.__all__:
        assert hasattr(pk, name), f"{name} not exposed on pycopykat.kernels"
