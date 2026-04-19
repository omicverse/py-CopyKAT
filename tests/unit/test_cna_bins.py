"""Tests for pycopykat.cna.bins — gene to 220 kb bin aggregation."""
import numpy as np
import pandas as pd

from pycopykat.cna.bins import aggregate_to_bins


def _gene_anno(chr_list, centers):
    half = 25
    return pd.DataFrame({
        "chromosome_name": chr_list,
        "start_position": [c - half for c in centers],
        "end_position":   [c + half for c in centers],
        "abspos":         centers,
    })


def test_medians_within_bin():
    # 4 genes on chrom 1 at centers 125, 225, 325, 425
    gene_anno = _gene_anno([1, 1, 1, 1], [125, 225, 325, 425])
    logCNA = np.array(
        [[1.0, 2.0],
         [2.0, 3.0],
         [3.0, 4.0],
         [4.0, 5.0]]
    )
    # bin schema: chrom + chrompos (end); start derives from previous chrompos
    bins = pd.DataFrame({"chrom": [1, 1], "chrompos": [300.0, 500.0], "abspos": [300, 500]})
    out = aggregate_to_bins(logCNA, gene_anno, bins, exclude_chrom_24=False)
    # bin 1 (0, 300]: genes 0,1 → medians [1,2]=1.5, [2,3]=2.5
    # bin 2 (300, 500]: genes 2,3 → medians [3,4]=3.5, [4,5]=4.5
    np.testing.assert_allclose(out[0], [1.5, 2.5])
    np.testing.assert_allclose(out[1], [3.5, 4.5])


def test_excludes_chrom_24_by_default():
    bins = pd.DataFrame({
        "chrom":     [1, 24, 24],
        "chrompos":  [500.0, 1000.0, 2000.0],
        "abspos":    [500, 1000, 2000],
    })
    gene_anno = _gene_anno([1, 1], [100, 400])
    logCNA = np.array([[1.0], [2.0]])
    out = aggregate_to_bins(logCNA, gene_anno, bins)
    assert out.shape == (1, 1)  # chrom 24 rows dropped


def test_missing_bin_forward_filled():
    gene_anno = _gene_anno([1, 1], [100, 400])
    logCNA = np.array([[1.0, 2.0], [3.0, 4.0]])
    # three bins, middle one has no genes → forward fill from earlier bin
    bins = pd.DataFrame({
        "chrom":    [1, 1, 1],
        "chrompos": [200.0, 300.0, 500.0],
        "abspos":   [200, 300, 500],
    })
    out = aggregate_to_bins(logCNA, gene_anno, bins, exclude_chrom_24=False)
    # bin 0 (0, 200]: gene 0 center=100 → [1, 2]
    # bin 1 (200, 300]: no gene → forward fill from bin 0 → [1, 2]
    # bin 2 (300, 500]: gene 1 center=400 → [3, 4]
    np.testing.assert_allclose(out[0], [1.0, 2.0])
    np.testing.assert_allclose(out[1], [1.0, 2.0])
    np.testing.assert_allclose(out[2], [3.0, 4.0])


def test_leading_missing_back_filled():
    gene_anno = _gene_anno([1], [400])
    logCNA = np.array([[7.0, 9.0]])
    # first bin empty, second bin holds the gene → leading NaN back-filled
    bins = pd.DataFrame({
        "chrom":    [1, 1],
        "chrompos": [200.0, 500.0],
        "abspos":   [200, 500],
    })
    out = aggregate_to_bins(logCNA, gene_anno, bins, exclude_chrom_24=False)
    np.testing.assert_allclose(out[0], [7.0, 9.0])
    np.testing.assert_allclose(out[1], [7.0, 9.0])
