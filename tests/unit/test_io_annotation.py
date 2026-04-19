"""Tests for pycopykat.io.annotation — hg20 gene annotation loader."""
import numpy as np
import pandas as pd

from pycopykat.io.annotation import (
    annotate_genes,
    load_hg20_annotation,
    load_hg20_bins,
    load_hg20_cycle_genes,
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


def test_annotate_genes_rejects_non_hg20():
    mat = pd.DataFrame(np.zeros((2, 1)), index=["TP53", "BRCA1"], columns=["c1"])
    try:
        annotate_genes(mat, id_type="Symbol", genome="hg19")
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError for genome='hg19'")
