"""Smoke tests for reference parquet / text files generated from copykat
sysdata.rda (see scripts/convert_sysdata.R for the canonical schema)."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

def test_hg20_gene_anno_parquet_exists():
    df = pd.read_parquet(DATA / "hg20_gene_anno.parquet")
    assert len(df) > 20_000
    for col in ("hgnc_symbol", "ensembl_gene_id", "chromosome_name",
                "start_position", "end_position", "abspos", "band"):
        assert col in df.columns, f"missing column {col}"

def test_hg20_bins_parquet_exists():
    df = pd.read_parquet(DATA / "hg20_220kb_bins.parquet")
    assert len(df) > 10_000
    assert {"chrom", "chrompos", "abspos"}.issubset(df.columns)
    # abspos is the genome-wide coordinate; must be monotonic for downstream segmentation
    assert df["abspos"].is_monotonic_increasing

def test_cyclegenes_txt():
    lines = (DATA / "hg20_cycle_genes.txt").read_text().strip().splitlines()
    # copykat 1.1.0 sysdata.rda ships 1316 cell-cycle genes; allow ≥ 1300 as a soft floor
    assert len(lines) >= 1_300
    # Verify they look like gene symbols (non-empty, no numeric factor levels leaked)
    assert all(s.strip() and not s.strip().isdigit() for s in lines[:10])
