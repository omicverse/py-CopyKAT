"""Tests for pycopykat.io.annotation — hg20 and mm10 gene annotation loaders."""
import numpy as np
import pandas as pd
import pytest

from pycopykat.io.annotation import (
    annotate_gene_names,
    annotate_genes,
    load_genome_annotation,
    load_hg20_annotation,
    load_hg20_bins,
    load_hg20_cycle_genes,
    load_mm10_annotation,
)


def test_load_returns_dataframe_with_expected_cols():
    ann = load_hg20_annotation()
    for col in (
        "hgnc_symbol",
        "ensembl_gene_id",
        "chromosome_name",
        "start_position",
        "end_position",
        "abspos",
    ):
        assert col in ann.columns


def test_load_cycle_genes_nonempty():
    genes = load_hg20_cycle_genes()
    assert len(genes) >= 1_300
    assert all(isinstance(g, str) and g for g in genes[:10])


def test_load_bins():
    bins = load_hg20_bins()
    assert {"chrom", "chrompos", "abspos"}.issubset(bins.columns)


def test_annotate_genes_symbol():
    ann = load_hg20_annotation()
    # pick 5 HGNC symbols that are not HLA- and not cell-cycle, to guarantee survival
    cyc = set(load_hg20_cycle_genes())
    candidates = ann["hgnc_symbol"].dropna().unique().tolist()
    syms = [s for s in candidates if not s.startswith("HLA-") and s not in cyc][:5]
    assert len(syms) == 5

    mat = pd.DataFrame(
        np.zeros((5, 3)), index=syms, columns=["c1", "c2", "c3"]
    )
    out = annotate_genes(mat, id_type="Symbol", genome="hg20")
    assert "chromosome_name" in out.columns
    assert "abspos" in out.columns
    # all 5 chosen symbols should survive
    assert out.shape[0] == 5
    # result must be sorted by abspos (ascending)
    assert out["abspos"].is_monotonic_increasing
    # cell columns must be preserved
    for c in ("c1", "c2", "c3"):
        assert c in out.columns


def test_annotate_genes_drops_hla_and_cycle():
    ann = load_hg20_annotation()
    cyc = load_hg20_cycle_genes()
    # find an HLA- and a cycle gene that exist in annotation
    hla_syms = [s for s in ann["hgnc_symbol"].dropna().unique() if s.startswith("HLA-")]
    cyc_in_ann = [s for s in cyc if s in set(ann["hgnc_symbol"].dropna())]
    assert hla_syms, "expected some HLA- symbols in hg20 annotation"
    assert cyc_in_ann, "expected some cycle genes to overlap hg20 annotation"

    keep_syms = (
        ann["hgnc_symbol"]
        .dropna()
        .loc[lambda s: ~s.str.startswith("HLA-") & ~s.isin(cyc)]
        .unique()[:2]
        .tolist()
    )
    syms = [hla_syms[0], cyc_in_ann[0], *keep_syms]
    mat = pd.DataFrame(np.zeros((len(syms), 1)), index=syms, columns=["c1"])
    out = annotate_genes(mat, id_type="Symbol", genome="hg20")
    assert hla_syms[0] not in set(out["hgnc_symbol"])
    assert cyc_in_ann[0] not in set(out["hgnc_symbol"])
    # only the two keep_syms should survive
    assert set(out["hgnc_symbol"]) == set(keep_syms)


def test_annotate_genes_rejects_unknown_genome():
    mat = pd.DataFrame(np.zeros((2, 1)), index=["TP53", "BRCA1"], columns=["c1"])
    with pytest.raises(NotImplementedError, match="hg19"):
        annotate_genes(mat, id_type="Symbol", genome="hg19")


# ---- mm10 / mouse ------------------------------------------------------


def test_load_mm10_annotation_schema():
    ann = load_mm10_annotation()
    # mm10 uses mgi_symbol (mouse gene names), not hgnc_symbol.
    for col in (
        "mgi_symbol",
        "ensembl_gene_id",
        "chromosome_name",
        "start_position",
        "end_position",
        "abspos",
    ):
        assert col in ann.columns, col
    assert "hgnc_symbol" not in ann.columns
    # R copykat ships 137 030 mm10 rows.
    assert len(ann) >= 130_000


def test_load_genome_annotation_dispatches():
    hg20 = load_genome_annotation("hg20")
    mm10 = load_genome_annotation("mm10")
    assert "hgnc_symbol" in hg20.columns
    assert "mgi_symbol" in mm10.columns
    with pytest.raises(NotImplementedError):
        load_genome_annotation("rn6")


def test_annotate_genes_mm10_symbol():
    ann = load_mm10_annotation()
    # Sample a few non-duplicated, non-null MGI symbols.
    syms = (
        ann["mgi_symbol"]
        .dropna()
        .drop_duplicates()
        .iloc[:5]
        .tolist()
    )
    assert len(syms) == 5

    mat = pd.DataFrame(np.zeros((5, 3)), index=syms, columns=["c1", "c2", "c3"])
    out = annotate_genes(mat, id_type="Symbol", genome="mm10")

    assert "mgi_symbol" in out.columns
    assert "chromosome_name" in out.columns
    assert "abspos" in out.columns
    assert out.shape[0] == 5
    assert out["abspos"].is_monotonic_increasing
    # cell columns must be preserved
    for c in ("c1", "c2", "c3"):
        assert c in out.columns


def test_annotate_genes_mm10_does_not_apply_hg20_filters():
    """R copykat's mm10 path skips HLA / cell-cycle filtering.

    If a user passes in an mm10 gene whose MGI symbol literally begins
    with ``HLA-`` (extremely unlikely but not forbidden) or coincides
    with a human cell-cycle gene name, mm10 must keep it. This is the
    behavioural difference from the hg20 code path.
    """
    ann = load_mm10_annotation()
    hg20_cycle = set(load_hg20_cycle_genes())
    # Find any mm10 symbols that happen to collide with hg20 cycle-gene names.
    mm10_syms = set(ann["mgi_symbol"].dropna().unique())
    overlap = sorted(mm10_syms & hg20_cycle)
    if not overlap:
        pytest.skip("no mm10/hg20-cycle symbol collision in this dataset")

    pick = overlap[0]
    mat = pd.DataFrame(np.zeros((1, 1)), index=[pick], columns=["c1"])
    out = annotate_genes(mat, id_type="Symbol", genome="mm10")
    assert pick in set(out["mgi_symbol"]), (
        "mm10 pipeline must NOT apply the hg20 cell-cycle filter"
    )


def test_annotate_gene_names_mm10_matches_annotate_genes():
    ann = load_mm10_annotation()
    all_syms = (
        ann.dropna(subset=["mgi_symbol"])["mgi_symbol"].drop_duplicates().tolist()
    )
    rng = np.random.default_rng(17)
    syms = rng.choice(all_syms, size=250, replace=False).tolist()
    syms = [*syms, "NOT_A_REAL_MOUSE_GENE"]
    mat = pd.DataFrame(np.zeros((len(syms), 2)), index=syms, columns=["c1", "c2"])

    out_full = annotate_genes(mat, id_type="Symbol", genome="mm10")
    gene_anno, row_idx = annotate_gene_names(syms, id_type="Symbol", genome="mm10")

    assert len(gene_anno) == len(out_full)
    assert gene_anno["mgi_symbol"].tolist() == out_full["mgi_symbol"].tolist()
    assert gene_anno["abspos"].is_monotonic_increasing
    recovered = [syms[i] for i in row_idx]
    assert recovered == gene_anno["mgi_symbol"].tolist()


def test_annotate_gene_names_matches_annotate_genes_ordering():
    """Sparse-path parity canary for the S1 patch.

    ``annotate_gene_names`` is used by the sparse pipeline to get the same
    post-HLA-and-cycle, abspos-sorted annotation that ``annotate_genes``
    produces for the DataFrame path. Any ordering drift here breaks the
    end-to-end `np.allclose(atol=1e-10)` gate on the sparse pipeline.
    """
    ann = load_hg20_annotation()
    # Sample a large mixed set: some HLA-, some cycle genes, some ordinary.
    all_syms = (
        ann.dropna(subset=["hgnc_symbol"])["hgnc_symbol"]
        .drop_duplicates()
        .tolist()
    )
    rng = np.random.default_rng(13)
    syms = rng.choice(all_syms, size=400, replace=False).tolist()
    # Also include a couple of genes not in the annotation to test the
    # drop-on-merge path.
    syms = [*syms, "NOT_A_REAL_GENE_42", "ZZZ_FAKE_XYZ"]
    mat = pd.DataFrame(
        np.zeros((len(syms), 2)), index=syms, columns=["c1", "c2"]
    )

    out_full = annotate_genes(mat, id_type="Symbol", genome="hg20")
    gene_anno, row_idx = annotate_gene_names(syms, id_type="Symbol", genome="hg20")

    # Row count + hgnc_symbol order must match bit-exactly.
    assert len(gene_anno) == len(out_full)
    assert gene_anno["hgnc_symbol"].tolist() == out_full["hgnc_symbol"].tolist()
    # abspos must be monotonic (sort is stable in annotate_gene_names too)
    assert gene_anno["abspos"].is_monotonic_increasing
    # row_idx must point to the same gene rows in the input order.
    recovered = [syms[i] for i in row_idx]
    assert recovered == gene_anno["hgnc_symbol"].tolist()
