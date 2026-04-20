"""Cell and gene QC filtering matching R copykat stage 1 (copykat.R ~L41-60).

Stage 1 drops low-complexity cells (fewer than ``min_gene_per_cell`` detected
genes) then drops genes with detection rate below ``low_dr`` across the
remaining cells.

The dense DataFrame / ndarray API is the original Phase-1 contract and is kept
bit-identical. Sparse sibling helpers (``filter_cells_and_genes_sparse``,
``filter_cells_by_chrom_coverage_sparse``) expose the same semantics on
``scipy.sparse`` inputs for the S1 sparse pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


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


# ─── Sparse sibling helpers (S1) ─────────────────────────────────────────────
# These operate on scipy.sparse matrices in genes × cells orientation and
# produce the same filtered outputs as the dense helpers above. The dense
# helpers above are kept untouched so the DataFrame public-API path is
# bit-identical with pre-S1.


def filter_cells_and_genes_sparse(
    X: sparse.spmatrix,
    gene_names: np.ndarray,
    cell_names: np.ndarray,
    *,
    min_gene_per_cell: int,
    low_dr: float,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, dict[str, int | str]]:
    """Sparse analog of :func:`filter_cells_and_genes`.

    Parameters
    ----------
    X
        Genes × cells sparse count matrix (any scipy.sparse format; internally
        converted once to CSC for column-nnz and once to CSR for row-nnz).
    gene_names, cell_names
        Axis labels aligned with ``X``'s rows / columns.
    min_gene_per_cell, low_dr
        Same thresholds as the dense helper. ``min_gene_per_cell`` uses a
        strict ``>`` comparison (matching the dense path); ``low_dr`` likewise.

    Returns
    -------
    (X_filtered, gene_names_out, cell_names_out, stats)
        ``X_filtered`` is CSR. ``stats`` carries the same keys as the dense
        helper.
    """
    if X.ndim != 2:
        raise ValueError(f"expected 2-D sparse matrix, got ndim={X.ndim}")
    if X.shape[0] != gene_names.shape[0]:
        raise ValueError(
            f"gene_names length {gene_names.shape[0]} != n_genes {X.shape[0]}"
        )
    if X.shape[1] != cell_names.shape[0]:
        raise ValueError(
            f"cell_names length {cell_names.shape[0]} != n_cells {X.shape[1]}"
        )

    # Column-nnz is O(nnz) on CSC.
    X_csc = X.tocsc()
    genes_per_cell = X_csc.getnnz(axis=0)  # (n_cells,)
    keep_cells = genes_per_cell > min_gene_per_cell
    n_cells_dropped = int((~keep_cells).sum())
    if keep_cells.sum() == 0:
        raise ValueError(
            f"no cells have more than {min_gene_per_cell} expressed genes"
        )

    # Slice out surviving cells; column slicing on CSC is efficient.
    X2_csc = X_csc[:, keep_cells]
    cell_names_out = np.asarray(cell_names)[keep_cells]

    # Gene detection rate: row-nnz on CSR.
    X2_csr = X2_csc.tocsr()
    n_cells_surv = X2_csr.shape[1]
    der = X2_csr.getnnz(axis=1) / n_cells_surv  # (n_genes,)
    keep_genes = der > low_dr

    # Row slicing on CSR is efficient; returned matrix stays CSR.
    X3 = X2_csr[keep_genes, :]
    gene_names_out = np.asarray(gene_names)[keep_genes]

    quality = "ok" if X3.shape[0] >= 7000 else "low"
    stats: dict[str, int | str] = {
        "n_cells_dropped": n_cells_dropped,
        "n_genes_kept": int(keep_genes.sum()),
        "data_quality": quality,
    }
    return X3, gene_names_out, cell_names_out, stats


def filter_cells_by_chrom_coverage_sparse(
    X: sparse.spmatrix,
    chrom: np.ndarray,
    *,
    ngene_chr: int,
) -> tuple[sparse.csc_matrix, list[int]]:
    """Sparse analog of :func:`filter_cells_by_chrom_coverage`.

    Uses the CSC ``indptr`` / ``indices`` directly: for each column we tally
    per-chrom non-zero rows via a C-level ``np.bincount``. The Python-level
    loop runs once per cell; each iteration is O(nnz_col) in C.

    Parameters
    ----------
    X
        Genes × cells sparse matrix (any format; internally converted to CSC).
    chrom
        Length-``n_genes`` integer chromosome ids aligned with ``X`` rows.
    ngene_chr
        Minimum non-zero genes required per chromosome per cell.

    Returns
    -------
    (X_filtered, dropped_cell_indices)
        ``X_filtered`` is CSC.
    """
    if X.shape[0] != chrom.shape[0]:
        raise ValueError(
            f"chrom length {chrom.shape[0]} != n_genes {X.shape[0]}"
        )

    X_csc = X.tocsc()
    n_cells = X_csc.shape[1]
    indptr = X_csc.indptr
    indices = X_csc.indices

    # Encode chrom as 0-based integer ids so np.bincount can use them directly.
    # Casting to intp avoids re-casting inside bincount for each cell.
    chrom_arr = np.asarray(chrom)
    _uniq, chrom_idx = np.unique(chrom_arr, return_inverse=True)
    chrom_idx = chrom_idx.astype(np.intp, copy=False)
    n_chr = int(_uniq.size)

    total_nz = np.diff(indptr)  # (n_cells,) per-cell nnz
    # Early reject for the "total_nz < 5" rule; still need per-chrom scan for
    # the rest to match dense semantics exactly.
    drop_mask = np.zeros(n_cells, dtype=bool)
    for j in range(n_cells):
        if total_nz[j] < 5:
            drop_mask[j] = True
            continue
        rows = indices[indptr[j]:indptr[j + 1]]
        per_chrom = np.bincount(chrom_idx[rows], minlength=n_chr)
        if (per_chrom < ngene_chr).any():
            drop_mask[j] = True

    dropped: list[int] = np.flatnonzero(drop_mask).tolist()
    keep_mask = ~drop_mask
    return X_csc[:, keep_mask], dropped
