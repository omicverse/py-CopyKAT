"""Reference tables of R copykat (``sysdata.rda``): gene annotations (hg20 / mm10), the 220 kb
genomic bins and the cell-cycle genes.

The parquet/text files under ``pycopykat/data/`` are written from copykat's ``sysdata.rda`` by
``scripts/convert_sysdata.R`` and are identical to R's tables. R copykat uses the hg20 bins for
mm10 as well.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

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
