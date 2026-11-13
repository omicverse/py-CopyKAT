"""The R pieces reproduced in :mod:`pycopykat._rcompat`, against values computed in R
(``tests/r_components.R`` -> ``tests/data/r_components.json``)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pycopykat._rcompat.hclust import hclust
from pycopykat._rcompat.mixtools import normalmix3
from pycopykat._rcompat.rbase import r_mean, r_sd
from pycopykat._rcompat.rgamma import GammaStream, rgamma_scalar
from pycopykat._rcompat.rng import RRandom
from pycopykat.core import euclidean_dist, ks_statistic

REF = json.loads((Path(__file__).parent / "data" / "r_components.json").read_text())


def test_rgamma_draws_identical_to_r():
    g = REF["rgamma"]
    rng = RRandom(1234)
    draws = GammaStream(rng).calls(g["shape"], g["rate"], 200, draws=True)
    assert np.array_equal(draws, np.array(g["draws"]))
    assert np.array_equal(rng.unif_rand(3), np.array(g["next_unif"]))     # same stream position


def test_rgamma_scalar_reference_identical_to_r():
    g = REF["rgamma"]
    rng = RRandom(1234)

    def unif():
        return float(rng.unif_rand(1)[0])

    for a, b, ref in zip(g["shape"], g["rate"], g["draws"]):
        assert [rgamma_scalar(unif, a, 1.0 / b) for _ in range(200)] == ref


def test_rgamma_means_identical_to_r():
    m = REF["means"]
    rng = RRandom(1234)
    got = GammaStream(rng, batch=64).calls(m["shape"], m["rate"], 1000)
    assert np.array_equal(got, np.array(m["mean"]))
    assert np.array_equal(rng.unif_rand(3), np.array(m["next_unif"]))


@pytest.mark.parametrize("batch", [1, 7, 128])
def test_rgamma_batching_does_not_change_draws(batch):
    rng0 = np.random.default_rng(0)
    shapes = np.concatenate([rng0.uniform(1, 500, 60), [0.7, 2.0, 3.5, 12.0]])
    rates = rng0.integers(1, 60, shapes.size).astype(float)
    ref = GammaStream(RRandom(42), batch=500).calls(shapes, rates, 300, draws=True)
    got = GammaStream(RRandom(42), batch=batch).calls(shapes, rates, 300, draws=True)
    assert np.array_equal(ref, got)


def test_rnorm_skips_zero_sd_as_r():
    r = REF["rnorm"]
    rng = RRandom(123)
    assert np.array_equal(rng.rnorm(np.zeros(len(r["sd"])), np.array(r["sd"])), np.array(r["x"]))


@pytest.mark.parametrize("method", ["ward.D", "ward.D2"])
def test_hclust_identical_to_r(method):
    h = REF["hclust"]
    ref = h[method]
    tree = hclust(euclidean_dist(np.array(h["x"]).T), method)
    assert np.array_equal(tree.merge, np.array(ref["merge"]))
    assert np.allclose(tree.height, ref["height"], rtol=1e-12, atol=0)
    assert np.array_equal(tree.order, np.array(ref["order"]))
    cut = np.array(ref["cut"])
    for j, k in enumerate(range(2, 7)):
        assert np.array_equal(tree.cutree(k), cut[:, j])


def test_normalmixem_matches_r():
    for e in REF["em"]:
        fit = normalmix3(np.array(e["x"])[None, :], np.array([e["sigma0"]]), maxit=5000)
        assert fit.iterations[0] == e["iterations"]
        assert fit.sigma[0] == pytest.approx(e["sigma"], rel=1e-12)
        assert np.allclose(fit.mu[0], e["mu"], rtol=1e-11, atol=1e-14)
        assert np.allclose(fit.lam[0], e["lambda"], rtol=1e-11)


def test_ks_statistic_matches_r():
    k = REF["ks"]
    assert ks_statistic(np.array(k["a"]), np.array(k["b"])) == k["stat"]
    assert ks_statistic(np.array(k["a2"]), np.array(k["b2"])) == k["stat2"]


def test_r_mean_and_sd():
    x = np.array(REF["ks"]["a"])
    assert r_mean(x) == pytest.approx(np.mean(x), rel=1e-15)
    assert r_sd(x) == pytest.approx(np.std(x, ddof=1), rel=1e-14)


def test_rgamma_per_call_fallback_identical(monkeypatch):
    """The per-call decoder (used when the region decoding does not settle) gives the same draws."""
    m = REF["means"]
    monkeypatch.setattr(GammaStream, "_batch_region", lambda self, *a, **k: None)
    rng = RRandom(1234)
    got = GammaStream(rng, batch=32).calls(m["shape"], m["rate"], 1000)
    assert np.array_equal(got, np.array(m["mean"]))
    assert np.array_equal(rng.unif_rand(3), np.array(m["next_unif"]))
