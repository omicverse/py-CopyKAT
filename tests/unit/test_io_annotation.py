"""Reference tables and R's gene annotation step (``annotateGenes.hg20``/``mm10`` + ``order``)."""
import numpy as np
import pytest

from pycopykat.core import annotate
from pycopykat.io.annotation import (
    load_genome_annotation,
    load_hg20_annotation,
    load_hg20_bins,
    load_hg20_cycle_genes,
    load_mm10_annotation,
)


def test_hg20_tables():
    ann = load_hg20_annotation()
    assert {"hgnc_symbol", "ensembl_gene_id", "chromosome_name", "start_position", "end_position",
            "abspos", "band"} <= set(ann.columns)
    assert len(ann) == 56051                      # R copykat's full.anno
    assert len(load_hg20_cycle_genes()) >= 1300
    bins = load_hg20_bins()
    assert list(bins.columns) == ["chrom", "chrompos", "abspos"] and len(bins) == 12205


def test_mm10_table_and_dispatch():
    mm = load_mm10_annotation()
    assert "mgi_symbol" in mm.columns and "hgnc_symbol" not in mm.columns and len(mm) == 137030
    assert "hgnc_symbol" in load_genome_annotation("hg20").columns
    with pytest.raises(NotImplementedError):
        load_genome_annotation("rn6")


def test_annotate_keeps_data_order_on_ties_and_sorts_by_abspos():
    ann = load_hg20_annotation()
    # two genes sharing an abspos: R orders by abspos stably after putting rows in data order
    dup = ann[ann.duplicated("abspos", keep=False)].sort_values("abspos").iloc[:2]["hgnc_symbol"].tolist()
    others = ann["hgnc_symbol"].iloc[1000:1003].tolist()
    genes = np.array([dup[1], "NOT_A_GENE", *others, dup[0]])
    out, rows = annotate(genes, "S", "hg20")
    assert out["abspos"].is_monotonic_increasing
    assert "NOT_A_GENE" not in set(out["hgnc_symbol"])
    pos = {g: i for i, g in enumerate(out["hgnc_symbol"])}
    assert pos[dup[1]] < pos[dup[0]]             # data order kept within the tie
    assert np.array_equal(genes[rows], out["hgnc_symbol"].to_numpy())


def test_annotate_ensembl_and_mm10():
    ann = load_hg20_annotation()
    ids = ann["ensembl_gene_id"].iloc[[500, 10, 3000]].to_numpy()
    out, rows = annotate(ids, "E", "hg20")
    assert np.array_equal(ids[rows], out["ensembl_gene_id"].to_numpy())
    mm = load_mm10_annotation()
    syms = mm["mgi_symbol"].dropna().drop_duplicates().iloc[:5].to_numpy()
    out, rows = annotate(syms, "S", "mm10")
    assert len(out) == 5 and out["abspos"].is_monotonic_increasing
