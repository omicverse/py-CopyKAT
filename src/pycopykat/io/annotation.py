"""hg20 gene annotation loader and annotate_genes helper.

Wraps the parquet reference files shipped under ``data/`` (generated from
copykat's ``sysdata.rda`` by ``scripts/convert_sysdata.R``). HLA- and
cell-cycle genes are removed during :func:`annotate_genes` to match R copykat
behaviour (copykat.R lines ~70-82).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


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


def annotate_gene_names(
    gene_names: "Sequence[str] | np.ndarray",
    *,
    id_type: str = "Symbol",
    genome: str = "hg20",
) -> tuple[pd.DataFrame, "np.ndarray"]:
    """Lookup hg20 coordinates for a bare gene-name list.

    Sparse-path analog of :func:`annotate_genes` that does NOT need the
    expression matrix attached. Returns the matched annotation DataFrame in
    the same row order as :func:`annotate_genes` produces, plus an integer
    index array mapping the annotated rows back to positions in the input
    ``gene_names``. The caller uses ``row_idx`` to row-slice the sparse
    counts matrix once.

    Parameters
    ----------
    gene_names
        Input gene identifiers aligned with the rows of the counts matrix.
    id_type
        ``"Symbol"`` or ``"Ensembl"``.
    genome
        Only ``"hg20"`` supported.

    Returns
    -------
    (gene_anno, row_idx)
        ``gene_anno`` is the annotation subset (HLA and cell-cycle dropped,
        sorted by ``abspos``). ``row_idx`` is an integer array with one
        entry per annotated row giving the corresponding row position in
        ``gene_names``.
    """
    import numpy as np

    if genome != "hg20":
        raise NotImplementedError(f"V1 only supports genome='hg20', got {genome!r}")
    if id_type not in ("Symbol", "Ensembl"):
        raise ValueError(f"id_type must be 'Symbol' or 'Ensembl', got {id_type!r}")

    ann = load_hg20_annotation()
    key = "hgnc_symbol" if id_type == "Symbol" else "ensembl_gene_id"
    ann = ann.dropna(subset=[key]).drop_duplicates(subset=[key])

    names = np.asarray(gene_names)
    # Use the same merge-then-filter order as annotate_genes so the resulting
    # row order matches bit-for-bit.
    expr_reset = pd.DataFrame(
        {key: names, "__row_idx": np.arange(names.size, dtype=np.int64)}
    )
    merged = ann.merge(expr_reset, on=key, how="inner")

    cyc = set(load_hg20_cycle_genes())
    is_hla = merged["hgnc_symbol"].astype(str).str.startswith("HLA-")
    is_cyc = merged["hgnc_symbol"].isin(cyc)
    merged = merged.loc[~(is_hla | is_cyc)].copy()

    merged = merged.sort_values("abspos", kind="mergesort").reset_index(drop=True)
    row_idx = merged["__row_idx"].to_numpy(dtype=np.int64)
    gene_anno = merged.drop(columns="__row_idx")
    return gene_anno, row_idx


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
