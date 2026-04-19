"""Integration test: segment_cells → aggregate_to_bins roundtrip."""
import numpy as np
import pandas as pd

from pycopykat.cna.bins import aggregate_to_bins
from pycopykat.segment.mcmc import segment_cells


def test_segment_then_bin_roundtrip():
    rng = np.random.default_rng(0)
    g, c = 400, 20
    fttmat = rng.standard_normal((g, c)) * 0.1
    clu = np.array([1] * 10 + [2] * 10)
    logCNA, BR = segment_cells(
        fttmat, clu, bins=25, ks_cut=0.2, seed=0, mc=200
    )

    # 400 genes evenly spaced on chrom 1, each 1 kb apart
    anno = pd.DataFrame({
        "chromosome_name": [1] * g,
        "start_position": np.arange(g) * 1000,
        "end_position": np.arange(g) * 1000 + 500,
        "abspos": np.arange(g) * 1000 + 250,
    })
    # 10 bins covering the gene range, each 40 kb wide (chrompos = end coord)
    bins = pd.DataFrame({
        "chrom":    [1] * 10,
        "chrompos": (np.arange(10) + 1) * 40_000.0,
        "abspos":   (np.arange(10) + 1) * 40_000.0,
    })
    out = aggregate_to_bins(logCNA, anno, bins, exclude_chrom_24=False)
    assert out.shape == (10, c)
    assert not np.isnan(out).any()
