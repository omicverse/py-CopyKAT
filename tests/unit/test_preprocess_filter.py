"""Tests for pycopykat.preprocess.filter — stage-1 QC and chrom coverage."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from pycopykat.preprocess.filter import (
    filter_cells_and_genes,
    filter_cells_and_genes_sparse,
    filter_cells_by_chrom_coverage,
    filter_cells_by_chrom_coverage_sparse,
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


# ─── sparse sibling tests ───────────────────────────────────────────────────


def _random_sparse_counts(
    n_genes: int, n_cells: int, density: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dense_int, sparse_int) poisson-sampled counts."""
    rng = np.random.default_rng(seed)
    mask = rng.random((n_genes, n_cells)) < density
    vals = rng.poisson(3, size=(n_genes, n_cells)).astype(np.int32)
    dense = np.where(mask, vals, 0)
    return dense, dense.copy()


def test_filter_cells_and_genes_sparse_matches_dense():
    dense, _ = _random_sparse_counts(200, 50, density=0.15, seed=42)
    genes = np.array([f"g{i}" for i in range(dense.shape[0])])
    cells = np.array([f"c{i}" for i in range(dense.shape[1])])

    df = pd.DataFrame(dense, index=genes, columns=cells)
    out_dense, stat_d = filter_cells_and_genes(
        df, min_gene_per_cell=5, low_dr=0.05
    )

    X_csr = sparse.csr_matrix(dense)
    X_out, g_out, c_out, stat_s = filter_cells_and_genes_sparse(
        X_csr, genes, cells, min_gene_per_cell=5, low_dr=0.05
    )

    # Shape + labels identical (same sort order — filter is deterministic).
    assert X_out.shape == out_dense.shape
    assert list(g_out) == out_dense.index.tolist()
    assert list(c_out) == out_dense.columns.tolist()
    # Values identical.
    np.testing.assert_array_equal(X_out.toarray(), out_dense.to_numpy())
    # Stats match.
    assert stat_s["n_cells_dropped"] == stat_d["n_cells_dropped"]
    assert stat_s["n_genes_kept"] == stat_d["n_genes_kept"]
    assert stat_s["data_quality"] == stat_d["data_quality"]


def test_filter_cells_and_genes_sparse_raises_when_all_cells_fail():
    X = sparse.csr_matrix(np.zeros((10, 3), dtype=np.int32))
    with pytest.raises(ValueError, match="no cells"):
        filter_cells_and_genes_sparse(
            X,
            np.array([f"g{i}" for i in range(10)]),
            np.array(["c1", "c2", "c3"]),
            min_gene_per_cell=1,
            low_dr=0.05,
        )


def test_filter_cells_by_chrom_coverage_sparse_matches_dense():
    rng = np.random.default_rng(7)
    # 3 chromosomes, unequal sizes; random 15% density.
    n_genes, n_cells = 90, 40
    chrom = np.array([1] * 30 + [2] * 35 + [3] * 25)
    mask = rng.random((n_genes, n_cells)) < 0.15
    vals = rng.poisson(4, size=(n_genes, n_cells)).astype(np.int32)
    dense = np.where(mask, vals, 0)

    out_d, dropped_d = filter_cells_by_chrom_coverage(
        dense, chrom, ngene_chr=2
    )
    X_csc = sparse.csc_matrix(dense)
    out_s, dropped_s = filter_cells_by_chrom_coverage_sparse(
        X_csc, chrom, ngene_chr=2
    )
    assert dropped_s == dropped_d
    assert out_s.shape == out_d.shape
    np.testing.assert_array_equal(out_s.toarray(), out_d)


def test_filter_cells_by_chrom_coverage_sparse_total_nz_rule():
    # 1 cell with 4 non-zero spread across 3 chromosomes — dropped even if
    # ngene_chr=1 would otherwise keep it (total_nz < 5 rule wins).
    chrom = np.array([1] * 10 + [2] * 10 + [3] * 10)
    X = np.zeros((30, 1), dtype=np.int32)
    X[[0, 10, 20, 11], 0] = 1
    out_s, dropped_s = filter_cells_by_chrom_coverage_sparse(
        sparse.csc_matrix(X), chrom, ngene_chr=1
    )
    assert dropped_s == [0]
    assert out_s.shape == (30, 0)


def test_copykat_from_sparse_matches_dense_tiny_pipeline():
    """End-to-end sparse vs dense parity on a tiny synthetic dataset.

    This is the S1 correctness canary: the sparse pipeline must produce
    bit-identical prediction and np.allclose CNA output vs the dense path.
    """
    import pandas as pd

    from pycopykat import CopykatConfig, copykat
    from pycopykat.io.annotation import load_hg20_annotation
    from pycopykat.pipeline import _copykat_from_sparse

    rng = np.random.default_rng(0)
    ann = load_hg20_annotation()
    syms_pool = (
        ann.dropna(subset=["hgnc_symbol"])
        .drop_duplicates("hgnc_symbol")
        .sort_values("abspos")
    )
    picks = (
        syms_pool.groupby("chromosome_name", sort=False)
        .head(80)
        .loc[syms_pool["chromosome_name"].isin(range(1, 11))]
    )
    genes = picks["hgnc_symbol"].unique().tolist()[:500]
    n_cells = 60
    # 20% dense so there are real zeros to exercise the sparse path.
    mask = rng.random((len(genes), n_cells)) < 0.6
    vals = rng.poisson(5, size=(len(genes), n_cells)).astype(np.int32)
    dense_mat = np.where(mask, vals, 0)
    cells = [f"c{i}" for i in range(n_cells)]

    cfg = CopykatConfig(
        min_gene_per_cell=10,
        low_dr=0.01,
        up_dr=0.02,
        win_size=10,
        ks_cut=0.3,
        ngene_chr=1,
        n_jobs=1,
        seed=1234,
        sam_name="tiny_parity",
    )
    df = pd.DataFrame(dense_mat, index=genes, columns=cells)
    res_dense = copykat(df, config=cfg)

    X_sparse = sparse.csr_matrix(dense_mat)
    res_sparse = _copykat_from_sparse(
        X_sparse, np.asarray(genes), np.asarray(cells), config=cfg
    )

    # Predictions identical cell-by-cell.
    pred_d = res_dense.prediction.set_index("cell")["copykat.pred"]
    pred_s = res_sparse.prediction.set_index("cell")["copykat.pred"]
    assert pred_d.index.equals(pred_s.index)
    pd.testing.assert_series_equal(pred_d, pred_s, check_names=False)

    # CNA matrix: np.allclose at tight tolerance.
    cna_d = res_dense.cna_mat.to_numpy()
    cna_s = res_sparse.cna_mat.to_numpy()
    assert cna_d.shape == cna_s.shape
    assert np.allclose(cna_d, cna_s, atol=1e-10, rtol=0)
