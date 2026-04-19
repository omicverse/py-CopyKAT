"""Tests for pycopykat.segment.breakpoint — KS-test breakpoint detection."""
import numpy as np

from pycopykat.segment.breakpoint import find_breakpoints


def test_detects_mean_shift():
    rng = np.random.default_rng(0)
    # 200-long signal: mean 1 for [0,100), mean 5 for [100,200)
    y = np.concatenate(
        [rng.poisson(1, size=100), rng.poisson(5, size=100)]
    ).astype(float)
    br = find_breakpoints(y, bins=25, ks_cut=0.1, seed=0)
    assert any(abs(b - 100) <= 25 for b in br), f"breaks={br}"


def test_returns_endpoints():
    y = np.ones(200)
    br = find_breakpoints(y, bins=25, ks_cut=0.1, seed=0)
    assert br[0] == 0
    assert br[-1] == 199


def test_short_signal_falls_back_to_endpoints_only():
    y = np.arange(40, dtype=float)
    br = find_breakpoints(y, bins=25, ks_cut=0.1, seed=0)
    assert br == [0, 39]


def test_sorted_and_unique():
    rng = np.random.default_rng(2)
    y = np.concatenate([rng.poisson(1, size=150), rng.poisson(10, size=150)]).astype(float)
    br = find_breakpoints(y, bins=25, ks_cut=0.1, seed=1)
    assert br == sorted(br)
    assert len(br) == len(set(br))
