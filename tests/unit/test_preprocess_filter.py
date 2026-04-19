"""Tests for pycopykat.preprocess.filter — stage-1 QC and chrom coverage."""
import numpy as np
import pandas as pd
import pytest

from pycopykat.preprocess.filter import (
    filter_cells_and_genes,
    filter_cells_by_chrom_coverage,
)


def test_filters_low_gene_cells():
    rng = np.random.default_rng(0)
    mat = pd.DataFrame(
        rng.poisson(5, size=(100, 4)),
        index=[f"g{i}" for i in range(100)],
        columns=["c1", "c2", "c3", "c4"],
    )
    # overwrite c1 so it expresses only 50 non-zero genes
    mat["c1"] = 0
    mat.iloc[:50, mat.columns.get_loc("c1")] = 1
    out, stats = filter_cells_and_genes(mat, min_gene_per_cell=60, low_dr=0.05)
    assert "c1" not in out.columns
    assert stats["n_cells_dropped"] == 1
    assert stats["data_quality"] in ("ok", "low")


def test_filters_low_dr_genes():
    rng = np.random.default_rng(1)
    mat = pd.DataFrame(
        rng.poisson(1, size=(50, 1000)),
        index=[f"g{i}" for i in range(50)],
        columns=[f"c{i}" for i in range(1000)],
    )
    mat.iloc[0, :] = 0  # g0 always zero → DR = 0 → dropped at low_dr=0.05
    out, stats = filter_cells_and_genes(mat, min_gene_per_cell=1, low_dr=0.05)
    assert "g0" not in out.index
    assert stats["n_genes_kept"] == out.shape[0]


def test_raises_when_no_cells_pass():
    mat = pd.DataFrame(
        np.zeros((10, 3), dtype=int),
        index=[f"g{i}" for i in range(10)],
        columns=["c1", "c2", "c3"],
    )
    with pytest.raises(ValueError, match="no cells"):
        filter_cells_and_genes(mat, min_gene_per_cell=1, low_dr=0.05)


def test_quality_flag_low_when_few_genes_remain():
    rng = np.random.default_rng(2)
    # deliberately few genes so quality should be "low"
    mat = pd.DataFrame(
        rng.poisson(5, size=(100, 10)),
        index=[f"g{i}" for i in range(100)],
        columns=[f"c{i}" for i in range(10)],
    )
    _, stats = filter_cells_and_genes(mat, min_gene_per_cell=1, low_dr=0.05)
    assert stats["data_quality"] == "low"


def test_chrom_coverage_requires_min_per_chrom():
    # 3 chroms × 10 genes each, 2 cells
    idx = pd.MultiIndex.from_arrays(
        [[1] * 10 + [2] * 10 + [3] * 10, list(range(30))],
        names=["chrom", "gene"],
    )
    mat = pd.DataFrame(np.ones((30, 2)), index=idx, columns=["c1", "c2"])
    # c2 has no expression on chrom 3
    mat.loc[(3, slice(None)), "c2"] = 0
    chrom = mat.index.get_level_values("chrom").to_numpy()
    out, dropped = filter_cells_by_chrom_coverage(mat.to_numpy(), chrom, ngene_chr=5)
    assert dropped == [1]  # c2 dropped (index 1)
    assert out.shape == (30, 1)


def test_chrom_coverage_drops_when_total_nonzero_below_5():
    # 3 chroms, 10 genes each, 1 cell with only 4 non-zero genes
    chrom = np.array([1] * 10 + [2] * 10 + [3] * 10)
    X = np.zeros((30, 1))
    X[[0, 10, 20, 11], 0] = 1  # 4 non-zero
    out, dropped = filter_cells_by_chrom_coverage(X, chrom, ngene_chr=1)
    assert dropped == [0]
    assert out.shape == (30, 0)


def test_chrom_coverage_keeps_cells_with_enough_per_chrom():
    chrom = np.array([1] * 10 + [2] * 10 + [3] * 10)
    X = np.ones((30, 2))  # every gene expressed in both cells
    out, dropped = filter_cells_by_chrom_coverage(X, chrom, ngene_chr=5)
    assert dropped == []
    assert out.shape == (30, 2)
