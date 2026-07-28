"""Gene annotation loaders (hg20 / mm10) and annotate_genes helpers.

Wraps the parquet reference files shipped inside the package under
``pycopykat/data/`` (generated from copykat's ``sysdata.rda`` by
``scripts/convert_sysdata.R``). For hg20, HLA-
and cell-cycle genes are removed during :func:`annotate_genes` to match R
copykat behaviour (copykat.R lines ~70-82). For mm10 the upstream R
implementation skips those filters and reuses the hg20 220 kb bin table;
pycopykat mirrors both choices.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


_GENOME_SYMBOL_COL = {
    "hg20": "hgnc_symbol",
    "mm10": "mgi_symbol",
}
_SUPPORTED_GENOMES = tuple(_GENOME_SYMBOL_COL)


#: Reference tables live inside the package (``pycopykat/data/``) so a wheel
#: install carries them. Up to 0.1.0.dev1 they sat at the repo root and were
#: resolved as ``parents[2] / "data"``, which only ever worked from a source
#: checkout — from ``site-packages`` that points at ``site-packages/data/``.
_PACKAGE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_LEGACY_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _data_dir() -> Path:
    """Return the directory holding the packaged reference tables."""
    if _PACKAGE_DATA_DIR.is_dir():
        return _PACKAGE_DATA_DIR
    if _LEGACY_DATA_DIR.is_dir():
        # Pre-0.1 source checkout with the tables still at the repo root.
        return _LEGACY_DATA_DIR
    raise FileNotFoundError(
        f"pycopykat reference data not found at {_PACKAGE_DATA_DIR}. The "
        "parquet/text tables ship inside the package; an installation that "
        "predates that move is missing them. Reinstall with:\n"
        "    pip install -U 'pycopykat @ git+https://github.com/omicverse/py-CopyKAT.git'"
    )


def load_hg20_annotation() -> pd.DataFrame:
    """Load the hg20 gene coordinate annotation as a DataFrame."""
    return pd.read_parquet(_data_dir() / "hg20_gene_anno.parquet")


def load_mm10_annotation() -> pd.DataFrame:
    """Load the mm10 gene coordinate annotation as a DataFrame.

    Mirrors R copykat's ``full.anno.mm10`` (137 030 rows). Uses
    ``mgi_symbol`` as the gene name key (vs ``hgnc_symbol`` for hg20);
    schema otherwise identical.
    """
    return pd.read_parquet(_data_dir() / "mm10_gene_anno.parquet")


def load_genome_annotation(genome: str) -> pd.DataFrame:
    """Load the gene-coordinate table for the requested genome."""
    if genome == "hg20":
        return load_hg20_annotation()
    if genome == "mm10":
        return load_mm10_annotation()
    raise NotImplementedError(
        f"Unsupported genome {genome!r}; expected one of {_SUPPORTED_GENOMES}"
    )


def load_hg20_cycle_genes() -> list[str]:
    """Return the list of cell-cycle gene symbols to exclude (hg20 only)."""
    text = (_data_dir() / "hg20_cycle_genes.txt").read_text()
    return [s.strip() for s in text.splitlines() if s.strip()]


def load_hg20_bins() -> pd.DataFrame:
    """Load the hg20 220 kb genomic bin table.

    R copykat also uses this bin table in the mm10 code path (no separate
    ``DNA.mm10`` in ``sysdata.rda``), so the loader name is genome-specific
    but the data is shared across both pipelines.
    """
    return pd.read_parquet(_data_dir() / "hg20_220kb_bins.parquet")


def _symbol_column(genome: str, id_type: str) -> str:
    """Pick the annotation key column for a (genome, id_type) combo."""
    if id_type == "Ensembl":
        return "ensembl_gene_id"
    if id_type != "Symbol":
        raise ValueError(f"id_type must be 'Symbol' or 'Ensembl', got {id_type!r}")
    try:
        return _GENOME_SYMBOL_COL[genome]
    except KeyError as exc:
        raise NotImplementedError(
            f"Unsupported genome {genome!r}; expected one of {_SUPPORTED_GENOMES}"
        ) from exc


def _apply_hg20_filters(merged: pd.DataFrame) -> pd.DataFrame:
    """Drop HLA- and cell-cycle genes (hg20-only, matches R copykat)."""
    cyc = set(load_hg20_cycle_genes())
    is_hla = merged["hgnc_symbol"].astype(str).str.startswith("HLA-")
    is_cyc = merged["hgnc_symbol"].isin(cyc)
    return merged.loc[~(is_hla | is_cyc)].copy()


def annotate_gene_names(
    gene_names: "Sequence[str] | np.ndarray",
    *,
    id_type: str = "Symbol",
    genome: str = "hg20",
) -> tuple[pd.DataFrame, "np.ndarray"]:
    """Lookup gene coordinates for a bare gene-name list.

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
        ``"hg20"`` (human, HGNC) or ``"mm10"`` (mouse, MGI).

    Returns
    -------
    (gene_anno, row_idx)
        ``gene_anno`` is the annotation subset (for hg20: HLA and cell-cycle
        dropped; for mm10: no extra filter, mirroring R copykat). Sorted by
        ``abspos``. ``row_idx`` is an integer array with one entry per
        annotated row giving the corresponding row position in
        ``gene_names``.
    """
    import numpy as np

    key = _symbol_column(genome, id_type)
    ann = load_genome_annotation(genome)
    ann = ann.dropna(subset=[key]).drop_duplicates(subset=[key])

    names = np.asarray(gene_names)
    # Use the same merge-then-filter order as annotate_genes so the resulting
    # row order matches bit-for-bit.
    expr_reset = pd.DataFrame(
        {key: names, "__row_idx": np.arange(names.size, dtype=np.int64)}
    )
    merged = ann.merge(expr_reset, on=key, how="inner")

    if genome == "hg20":
        merged = _apply_hg20_filters(merged)

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
    """Attach gene coordinates to a genes × cells expression DataFrame.

    Parameters
    ----------
    expr
        Genes × cells DataFrame. Index holds gene identifiers.
    id_type
        ``"Symbol"`` (HGNC for hg20 / MGI for mm10) or ``"Ensembl"``.
    genome
        ``"hg20"`` (human) or ``"mm10"`` (mouse).

    Returns
    -------
    pd.DataFrame
        Rows = genes that mapped to an annotation entry. For hg20, HLA-
        and cell-cycle genes are additionally dropped. For mm10 no extra
        filter is applied (matches R copykat's mm10 code path). Columns =
        all 7 annotation columns followed by the original cell columns.
        Sorted by ``abspos`` ascending.
    """
    key = _symbol_column(genome, id_type)
    ann = load_genome_annotation(genome)
    ann = ann.dropna(subset=[key]).drop_duplicates(subset=[key])

    expr_reset = expr.rename_axis(index=key).reset_index()
    merged = ann.merge(expr_reset, on=key, how="inner")

    if genome == "hg20":
        merged = _apply_hg20_filters(merged)

    merged = merged.sort_values("abspos", kind="mergesort").reset_index(drop=True)
    return merged
