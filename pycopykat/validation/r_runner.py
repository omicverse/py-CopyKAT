"""Subprocess wrapper around the canonical R copykat driver.

Produces the same output files the R package emits so comparison utilities
(:mod:`pycopykat.validation.metrics`) can read them directly.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


_DEFAULT_DRIVER = Path(__file__).resolve().parents[2] / "scripts" / "run_r_copykat.R"


def run_r_copykat(
    mat: pd.DataFrame,
    out_dir: Path,
    *,
    sam_name: str,
    n_cores: int = 1,
    driver: Path | None = None,
    counts_path: Path | None = None,
) -> Path:
    """Invoke R copykat on ``mat`` via ``scripts/run_r_copykat.R``.

    Parameters
    ----------
    mat
        genes × cells DataFrame with HGNC symbols as index.
    out_dir
        Directory to receive copykat's ``<sam>_copykat_*.txt`` files.
    sam_name
        Sample name prefix for output files.
    n_cores
        Cores for copykat's internal mclapply.
    driver
        Override the R driver path.
    counts_path
        Where to stage the counts TSV (default ``out_dir / "counts.tsv"``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    driver = Path(driver) if driver else _DEFAULT_DRIVER
    counts_path = Path(counts_path) if counts_path else out_dir / "counts.tsv"
    mat.to_csv(counts_path, sep="\t")
    cmd = ["Rscript", str(driver), str(counts_path), str(out_dir), sam_name, str(n_cores)]
    subprocess.run(cmd, check=True)
    return out_dir


def load_r_prediction(out_dir: Path, sam_name: str) -> pd.DataFrame:
    """Load the R copykat prediction file (cell × copykat.pred)."""
    path = Path(out_dir) / f"{sam_name}_copykat_prediction.txt"
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] != 2:
        raise ValueError(f"expected 2 cols in {path}, got {df.shape[1]}")
    df.columns = ["cell", "copykat.pred"]
    return df


def load_r_cna(out_dir: Path, sam_name: str) -> pd.DataFrame:
    """Load the R copykat CNA results file with (chrom, chrompos, abspos) as index."""
    path = Path(out_dir) / f"{sam_name}_copykat_CNA_results.txt"
    df = pd.read_csv(path, sep="\t")
    # R writes first 3 cols = chrom, chrompos, abspos; remainder = cells
    meta = df.columns[:3].tolist()
    df = df.set_index(meta)
    return df
