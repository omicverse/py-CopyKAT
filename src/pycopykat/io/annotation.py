"""hg20 gene annotation loader and annotate_genes helper.

Wraps the parquet reference files shipped under ``data/`` (generated from
copykat's ``sysdata.rda`` by ``scripts/convert_sysdata.R``). HLA- and
cell-cycle genes are removed during :func:`annotate_genes` to match R copykat
behaviour (copykat.R lines ~70-82).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _data_dir() -> Path:
    """Return the project-level ``data/`` directory.

    Resolution assumes the installed layout ``<root>/src/pycopykat/io/annotation.py``
    so that ``parents[3]`` is the project root. With ``pip install -e .`` this
    works; for a wheel install the same parquet files should be shipped via
    package data — V1 sticks with the editable layout.
    """
    return Path(__file__).resolve().parents[3] / "data"


def load_hg20_annotation() -> pd.DataFrame:
    """Load the hg20 gene coordinate annotation as a DataFrame."""
    return pd.read_parquet(_data_dir() / "hg20_gene_anno.parquet")


def load_hg20_cycle_genes() -> list[str]:
    """Return the list of cell-cycle gene symbols to exclude."""
    text = (_data_dir() / "hg20_cycle_genes.txt").read_text()
    return [s.strip() for s in text.splitlines() if s.strip()]


def load_hg20_bins() -> pd.DataFrame:
    """Load the hg20 220 kb genomic bin table."""
    return pd.read_parquet(_data_dir() / "hg20_220kb_bins.parquet")


def annotate_genes(
    expr: pd.DataFrame,
    *,
    id_type: str = "Symbol",
    genome: str = "hg20",
) -> pd.DataFrame:
    """Attach hg20 coordinates to a genes × cells expression DataFrame.

    Parameters
    ----------
    expr
        Genes × cells DataFrame. Index holds gene identifiers.
    id_type
        ``"Symbol"`` (HGNC) or ``"Ensembl"``.
    genome
        Only ``"hg20"`` is supported in V1.

    Returns
    -------
    pd.DataFrame
        Rows = genes that mapped to an annotation entry, minus HLA- and cell-cycle
        genes. Columns = all 7 annotation columns followed by the original cell
        columns. Sorted by ``abspos`` ascending.
    """
    if genome != "hg20":
        raise NotImplementedError(f"V1 only supports genome='hg20', got {genome!r}")
    if id_type not in ("Symbol", "Ensembl"):
        raise ValueError(f"id_type must be 'Symbol' or 'Ensembl', got {id_type!r}")

    ann = load_hg20_annotation()
    key = "hgnc_symbol" if id_type == "Symbol" else "ensembl_gene_id"
    ann = ann.dropna(subset=[key]).drop_duplicates(subset=[key])

    expr_reset = expr.rename_axis(index=key).reset_index()
    merged = ann.merge(expr_reset, on=key, how="inner")

    cyc = set(load_hg20_cycle_genes())
    is_hla = merged["hgnc_symbol"].astype(str).str.startswith("HLA-")
    is_cyc = merged["hgnc_symbol"].isin(cyc)
    merged = merged.loc[~(is_hla | is_cyc)].copy()

    merged = merged.sort_values("abspos", kind="mergesort").reset_index(drop=True)
    return merged
