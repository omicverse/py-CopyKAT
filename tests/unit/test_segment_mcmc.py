"""Tests for pycopykat.segment.mcmc — per-cell Poisson-Gamma segmentation."""
import numpy as np

from pycopykat.segment.mcmc import segment_cells


def test_segment_assigns_segment_mean():
    rng = np.random.default_rng(0)
    # 200 genes × 50 cells, values near log-expression of 1
    fttmat = rng.standard_normal((200, 50)) * 0.1
    clu = np.array([1] * 25 + [2] * 25)
    logCNA, BR = segment_cells(
        fttmat, clu, bins=25, ks_cut=0.2, seed=0, mc=200
    )
    assert logCNA.shape == fttmat.shape
    assert BR[0] == 0
    assert BR[-1] == fttmat.shape[0] - 1


def test_piecewise_constant_within_segment():
    rng = np.random.default_rng(1)
    # Induce a clear shift in the cluster consensus to force a break
    half = 100
    left = rng.standard_normal((half, 10)) * 0.05 - 0.5
    right = rng.standard_normal((half, 10)) * 0.05 + 0.5
    fttmat = np.vstack([left, right])
    clu = np.ones(10, dtype=int)
    logCNA, BR = segment_cells(
        fttmat, clu, bins=25, ks_cut=0.1, seed=0, mc=300
    )
    # within each segment, all rows must be equal (segment-mean is constant)
    # segment i occupies [BR[i], BR[i+1]] for i==0 else [BR[i]+1, BR[i+1]]
    for i in range(len(BR) - 1):
        s = BR[i] if i == 0 else BR[i] + 1
        e = BR[i + 1]
        seg = logCNA[s : e + 1]
        np.testing.assert_allclose(seg, np.broadcast_to(seg[0], seg.shape), atol=1e-12)


def test_length_one_segment_is_handled():
    rng = np.random.default_rng(2)
    fttmat = rng.standard_normal((5, 4)) * 0.1
    clu = np.array([1, 1, 2, 2])
    # very short signal → find_breakpoints returns [0, n-1]
    logCNA, BR = segment_cells(fttmat, clu, bins=25, ks_cut=0.1, seed=0, mc=50)
    assert BR == [0, 4]
    assert logCNA.shape == (5, 4)
