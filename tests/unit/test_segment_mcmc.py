"""Tests for pycopykat.segment.mcmc — per-cell Poisson-Gamma segmentation."""
import numpy as np

from pycopykat.segment.mcmc import segment_cells


def _naive_analytical_segment(
    raw: np.ndarray, BR: list[int]
) -> np.ndarray:
    """Reference implementation: the exact per-cell Python loop that A1's
    ``_segment_one_cell`` performed with the analytical posterior mean.

    Segment i occupies ``[BR[i], BR[i+1]]`` for ``i == 0``, and
    ``[BR[i]+1, BR[i+1]]`` inclusive for ``i >= 1``. The value assigned to
    each gene in the segment is ``(max(mean, 1e-3) + sum) / (1 + len)``.
    """
    n_genes, n_cells = raw.shape
    x = np.empty((n_genes, n_cells), dtype=np.float64)
    for j in range(n_cells):
        col = raw[:, j]
        for i in range(len(BR) - 1):
            s = BR[i] if i == 0 else BR[i] + 1
            e = BR[i + 1]
            seg = col[s : e + 1]
            a = max(float(seg.mean()), 1e-3)
            x[s : e + 1, j] = (a + float(seg.sum())) / (1.0 + seg.size)
    return np.log(np.maximum(x, 1e-12))


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


def test_vectorized_matches_per_cell_analytical_reference():
    """A2 equivalence gate: the vectorized segmentation must produce bit-for-bit
    (up to float summation order) identical output to the naive per-cell
    Python loop computing the analytical Poisson-Gamma posterior mean.
    """
    rng = np.random.default_rng(20260420)
    # Synthetic (VST-like) matrix with a clear mid-signal shift to induce
    # a non-trivial shared breakpoint set.
    n_genes, n_cells = 400, 60
    left = rng.standard_normal((n_genes // 2, n_cells)) * 0.08 - 0.4
    right = rng.standard_normal((n_genes - n_genes // 2, n_cells)) * 0.08 + 0.4
    fttmat = np.vstack([left, right])
    clu = np.array([1] * (n_cells // 2) + [2] * (n_cells - n_cells // 2))

    logCNA, BR = segment_cells(
        fttmat, clu, bins=25, ks_cut=0.1, seed=0, mc=200
    )
    raw = np.exp(np.asarray(fttmat, dtype=np.float64))
    ref = _naive_analytical_segment(raw, BR)
    np.testing.assert_allclose(logCNA, ref, atol=1e-12, rtol=0)
