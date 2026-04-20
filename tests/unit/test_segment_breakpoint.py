"""Tests for pycopykat.segment.breakpoint — KS-test breakpoint detection."""
import numpy as np
from scipy.stats import ks_2samp

from pycopykat.config import CopykatConfig
from pycopykat.segment.breakpoint import (
    find_breakpoints,
    find_breakpoints_analytic,
    ks_gamma_analytic,
)


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


# ---------------------------------------------------------------------------
# B1 (scaffold): analytic Gamma-KS path
# ---------------------------------------------------------------------------


def test_ks_gamma_analytic_matches_ks2samp():
    """Analytic KS must agree with ``ks_2samp`` on 100k samples within 0.005.

    Exercises three (shape, scale) pairs — similar, dissimilar, and one near
    the small-shape degenerate corner — so we probe both numerical regimes.
    """
    pairs = [
        # (shape1, scale1, shape2, scale2)
        (2.0, 1.0, 10.0, 1.0),
        (25.0, 0.04, 30.0, 0.04),
        (1e-3, 1.0, 5.0, 1.0),
    ]
    rng = np.random.default_rng(20260420)
    for s1, sc1, s2, sc2 in pairs:
        d1 = rng.gamma(shape=s1, scale=sc1, size=100_000)
        d2 = rng.gamma(shape=s2, scale=sc2, size=100_000)
        mc_stat = float(ks_2samp(d1, d2).statistic)
        an_stat = ks_gamma_analytic(s1, sc1, s2, sc2)
        assert abs(mc_stat - an_stat) < 0.005, (
            f"pair={(s1, sc1, s2, sc2)}: mc={mc_stat:.4f} analytic={an_stat:.4f}"
        )


def test_ks_gamma_analytic_identical_is_zero():
    assert ks_gamma_analytic(5.0, 1.0, 5.0, 1.0) < 1e-6


def test_find_breakpoints_analytic_detects_step_change():
    """Step change at index 200 must be detected (within one bin width)."""
    rng = np.random.default_rng(42)
    y = np.concatenate(
        [rng.gamma(2.0, 1.0, size=200), rng.gamma(10.0, 1.0, size=200)]
    )
    br = find_breakpoints_analytic(y, bins=25, ks_cut=0.1)
    assert any(abs(b - 200) <= 25 for b in br), f"breaks={br}"


def test_find_breakpoints_analytic_endpoints_only_on_flat():
    """Flat signal → only endpoints returned."""
    y = np.ones(200, dtype=float)
    br = find_breakpoints_analytic(y, bins=25, ks_cut=0.1)
    assert br[0] == 0
    assert br[-1] == 199


def test_find_breakpoints_analytic_short_signal_fallback():
    y = np.arange(40, dtype=float)
    br = find_breakpoints_analytic(y, bins=25, ks_cut=0.1)
    assert br == [0, 39]


def test_default_config_keeps_mc_path():
    """Defensive test guarding the default. If this ever flips, the
    regression suite and R-parity ARI baseline must be re-validated."""
    assert CopykatConfig().breakpoint_ks == "mc"
