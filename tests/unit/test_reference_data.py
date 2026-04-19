# Tests for converted sysdata.rda reference files.
#
# Canonical column names per sysdata.rda full.anno schema as of copykat 1.1.0:
#   full.anno columns: abspos, chromosome_name, start_position, end_position,
#                      ensembl_gene_id, hgnc_symbol, band
#   DNA.hg20 columns:  chrom, chrompos, abspos
#   cyclegenes:        data.frame with 1 factor column "x" (1316 genes)
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def test_hg20_gene_anno_parquet_exists():
    df = pd.read_parquet(DATA / "hg20_gene_anno.parquet")
    assert len(df) > 20_000
    for col in (
        "hgnc_symbol",
        "chromosome_name",
        "start_position",
        "end_position",
        "abspos",
    ):
        assert col in df.columns, f"missing column {col}"


def test_hg20_bins_parquet_exists():
    df = pd.read_parquet(DATA / "hg20_220kb_bins.parquet")
    assert len(df) > 10_000
    assert {"chrom", "abspos"}.issubset(df.columns)


def test_cyclegenes_txt():
    lines = (DATA / "hg20_cycle_genes.txt").read_text().strip().splitlines()
    assert len(lines) > 100
