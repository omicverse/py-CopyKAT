"""Cell and gene QC filtering matching R copykat stage 1 (copykat.R ~L41-60).

Stage 1 drops low-complexity cells (fewer than ``min_gene_per_cell`` detected
genes) then drops genes with detection rate below ``low_dr`` across the
remaining cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def filter_cells_and_genes(
    mat: pd.DataFrame,
    *,
    min_gene_per_cell: int,
    low_dr: float,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Drop low-complexity cells and low-detection-rate genes.

    Parameters
    ----------
    mat
        Genes × cells raw count DataFrame.
    min_gene_per_cell
        Cells with ``(counts > 0).sum() <= min_gene_per_cell`` are dropped.
    low_dr
        Genes with detection rate ``(counts > 0).mean() <= low_dr`` across the
        surviving cells are dropped.

    Returns
    -------
    (filtered_mat, stats)
        ``filtered_mat`` is the genes × cells subset. ``stats`` holds the keys
        ``n_cells_dropped``, ``n_genes_kept`` and ``data_quality``
        (``"ok"`` if ≥ 7000 genes survive, otherwise ``"low"`` — matches R
        copykat's quality heuristic).
    """
    X = mat.to_numpy()
    genes_per_cell = (X > 0).sum(axis=0)
    keep_cells = genes_per_cell > min_gene_per_cell
    n_cells_dropped = int((~keep_cells).sum())
    if keep_cells.sum() == 0:
        raise ValueError(
            f"no cells have more than {min_gene_per_cell} expressed genes"
        )

    mat2 = mat.loc[:, keep_cells]
    X2 = mat2.to_numpy()
    der = (X2 > 0).sum(axis=1) / X2.shape[1]
    keep_genes = der > low_dr
    mat3 = mat2.loc[keep_genes]

    quality = "ok" if mat3.shape[0] >= 7000 else "low"
    stats: dict[str, int | str] = {
        "n_cells_dropped": n_cells_dropped,
        "n_genes_kept": int(keep_genes.sum()),
        "data_quality": quality,
    }
    return mat3, stats


def filter_cells_by_chrom_coverage(
    X: np.ndarray,
    chrom: np.ndarray,
    *,
    ngene_chr: int,
) -> tuple[np.ndarray, list[int]]:
    """Secondary cell filter requiring per-chromosome coverage.

    Matches copykat.R lines ~84-96. A cell is dropped if either:

    * the total number of non-zero genes is < 5, or
    * any chromosome has fewer than ``ngene_chr`` non-zero genes.

    Parameters
    ----------
    X
        Genes × cells expression array, rows already sorted by abspos.
    chrom
        Length-``n_genes`` array of chromosome ids aligned with ``X`` rows.
    ngene_chr
        Minimum non-zero genes required per chromosome per cell.

    Returns
    -------
    (filtered_X, dropped_cell_indices)
    """
    # Vectorized: permute rows by chrom id, then per-cell per-chrom nonzero
    # counts fall out of a single np.add.reduceat call on the boolean matrix.
    n_cells = X.shape[1]
    nz = X > 0  # (n_genes, n_cells) bool
    total_nz = nz.sum(axis=0)  # per-cell total

    _unique_chroms, chrom_idx = np.unique(chrom, return_inverse=True)
    order = np.argsort(chrom_idx, kind="stable")
    nz_ord = nz[order]
    chrom_ord = chrom_idx[order]
    # seg_starts[i] = first row in nz_ord belonging to chrom id i
    seg_starts = np.searchsorted(chrom_ord, np.arange(_unique_chroms.size))
    per_chrom_nz = np.add.reduceat(nz_ord, seg_starts, axis=0)
    # (n_chr, n_cells); cells with any chrom below threshold are dropped
    drop_mask = (total_nz < 5) | (per_chrom_nz < ngene_chr).any(axis=0)
    dropped: list[int] = np.flatnonzero(drop_mask).tolist()

    keep_mask = ~drop_mask
    return X[:, keep_mask], dropped
