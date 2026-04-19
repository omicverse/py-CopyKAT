"""Tests for pycopykat.baseline.gmm — per-cell GMM fallback."""
import numpy as np
import pandas as pd

from pycopykat.baseline._shared import BaselineResult
from pycopykat.baseline.gmm import baseline_gmm


def test_classifies_diploid_cells_from_tight_noise():
    rng = np.random.default_rng(0)
    g = 800
    # 20 extremely tight diploid cells (noise << mu_cut so all mass at |mu|<=0.05)
    dip = rng.normal(0, 0.015, size=(g, 20))
    # 20 aneuploid-like cells with broad/shifted distribution
    ane = rng.normal(0.3, 0.25, size=(g, 20))
    X = np.hstack([dip, ane])
    names = [f"d{i}" for i in range(20)] + [f"a{i}" for i in range(20)]

    res = baseline_gmm(
        X, cell_names=names, max_normal=5, mu_cut=0.05, nfraq_cut=0.9, seed=0,
    )
    assert isinstance(res, BaselineResult)
    # Early-stop should take the first few (all diploid)
    assert 0 < len(res.preN) <= 5
    n_d = sum(1 for n in res.preN if n.startswith("d"))
    # All collected diploids must come from the genuinely-diploid block
    assert n_d == len(res.preN), f"found non-diploid in preN: {res.preN}"
    # basel is mean of preN cells → expect very near zero
    assert res.basel.shape == (g,)
    assert np.abs(res.basel).mean() < 0.02


def test_returns_fallback_when_too_few_diploids():
    rng = np.random.default_rng(1)
    g = 300
    # All cells are aneuploid-like → <3 diploid detected
    X = rng.normal(0.5, 0.4, size=(g, 30))
    names = [f"c{i}" for i in range(30)]

    sentinel = BaselineResult(
        basel=np.full(g, -9.0), preN=["sentinel"],
        warning="fallback_used", labels=np.ones(30, dtype=int),
    )
    res = baseline_gmm(X, cell_names=names, max_normal=5, seed=0, fallback=sentinel)
    assert res is sentinel


def test_no_fallback_returns_warning_when_too_few_diploids():
    rng = np.random.default_rng(2)
    X = rng.normal(0.5, 0.4, size=(200, 20))
    res = baseline_gmm(X, max_normal=5, seed=0)
    assert res.warning == "unclassified.prediction"
    assert res.preN == []
