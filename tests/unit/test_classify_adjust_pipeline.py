"""Tests for pycopykat.classify.adjust_pipeline — pre-final-classification adjust."""
import numpy as np

from pycopykat.classify.adjust_pipeline import baseline_adjust


def test_adjust_mutes_values_within_sd_window():
    rng = np.random.default_rng(0)
    cna = rng.standard_normal((100, 20)) * 0.1
    diploid_mask = np.zeros(20, dtype=bool)
    diploid_mask[:10] = True

    out = baseline_adjust(cna, diploid_mask, factor=0.25)
    # diploid columns should shrink after muting
    assert out[:, diploid_mask].std() < cna[:, diploid_mask].std()
    assert out.shape == cna.shape


def test_requires_at_least_two_diploid_cells():
    rng = np.random.default_rng(1)
    cna = rng.standard_normal((50, 5)) * 0.1
    mask = np.array([True, False, False, False, False])
    try:
        baseline_adjust(cna, mask)
    except ValueError:
        return
    raise AssertionError("expected ValueError when <2 diploid cells")


def test_columns_mean_zero_after_recenter():
    rng = np.random.default_rng(2)
    cna = rng.standard_normal((200, 30)) * 0.2
    mask = np.zeros(30, dtype=bool)
    mask[:15] = True
    out = baseline_adjust(cna, mask, factor=0.25)
    # The final step re-centers columns at 0
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)
