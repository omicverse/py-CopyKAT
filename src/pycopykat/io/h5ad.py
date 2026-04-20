"""AnnData (h5ad) batch-split driver for :func:`pycopykat.copykat`.

Given a cells × genes :class:`anndata.AnnData` with a per-cell batch label in
``obs[batch_key]``, :func:`copykat_by_batch` runs the full copykat pipeline
independently on each batch and merges per-cell predictions + per-cell CNA
vectors back into the AnnData.

The per-batch behaviour matches the standard recommendation from the original
R copykat paper (Gao et al. 2021, *Nat. Biotech.*), which emphasises that the
baseline-cluster selection and diploid/aneuploid call should be estimated
within a single patient/library to avoid cross-patient batch effects. See the
R driver comments in ``copykat-R/R/copykat.R`` for the equivalent single-sample
assumption.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse

from pycopykat.config import CopykatConfig
from pycopykat.io.annotation import load_hg20_bins
from pycopykat.pipeline import _copykat_from_sparse, copykat

if TYPE_CHECKING:
    from anndata import AnnData


@dataclass(slots=True)
class _SparseBatch:
    """Per-batch sparse payload: genes × cells CSC with string axis labels."""
    X: sparse.spmatrix  # genes × cells
    gene_names: np.ndarray
    cell_names: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.X.shape[0]), int(self.X.shape[1]))

_PRED_COL = "copykat_pred"
_SUBCLONE_COL = "copykat_subclone"
_BATCH_COL = "copykat_batch"
_WARN_COL = "copykat_warnings"
_CNA_KEY = "copykat_cna"
_BINS_KEY = "copykat_bins"
_META_KEY = "copykat_per_batch"


def _extract_counts_frame(
    adata: "AnnData", cells: np.ndarray, layer: str | None
) -> "pd.DataFrame | _SparseBatch":
    """Return a genes × cells batch payload for the given cell mask.

    When the source AnnData matrix is sparse, keep it sparse (CSC, since
    genes × cells means cells are columns) and return a ``_SparseBatch`` so
    the pipeline can run step-1..3 on the sparse matrix. Dense AnnData .X
    still produces a pandas DataFrame to preserve the original public code
    path bit-identically.
    """
    sub = adata[cells, :]
    X = sub.layers[layer] if layer is not None else sub.X
    gene_names = np.asarray(sub.var_names.astype(str).to_numpy())
    cell_names = np.asarray(sub.obs_names.astype(str).to_numpy())
    if sparse.issparse(X):
        # AnnData stores cells × genes. Transpose once to genes × cells; CSR.T
        # is CSC for free (same buffers, no copy), which is what we want for
        # column-oriented cell slicing downstream.
        X_T = X.T
        if not sparse.issparse(X_T):
            X_T = sparse.csc_matrix(X_T)
        return _SparseBatch(X=X_T, gene_names=gene_names, cell_names=cell_names)
    # Dense fallback — preserves the original DataFrame path exactly.
    X = np.asarray(X, dtype=np.float64)
    return pd.DataFrame(X.T, index=gene_names, columns=cell_names)


def _run_one_batch(
    batch_label: str,
    counts: "pd.DataFrame | _SparseBatch",
    config: CopykatConfig,
) -> dict[str, object]:
    """Run the copykat pipeline for one batch and return a small payload dict."""
    t0 = time.time()
    cfg = CopykatConfig(
        id_type=config.id_type,
        genome=config.genome,
        cell_line=config.cell_line,
        ngene_chr=config.ngene_chr,
        min_gene_per_cell=config.min_gene_per_cell,
        low_dr=config.low_dr,
        up_dr=config.up_dr,
        win_size=config.win_size,
        ks_cut=config.ks_cut,
        distance=config.distance,
        norm_cell_names=config.norm_cell_names,
        sam_name=f"{config.sam_name}_{batch_label}" if config.sam_name else batch_label,
        output_dir=config.output_dir,
        output_seg=False,
        plot_genes=False,
        n_jobs=config.n_jobs,
        seed=config.seed,
        backend=config.backend,
    )
    if isinstance(counts, _SparseBatch):
        res = _copykat_from_sparse(
            counts.X, counts.gene_names, counts.cell_names, config=cfg
        )
        n_cells_in = counts.shape[1]
    else:
        res = copykat(counts, config=cfg)
        n_cells_in = counts.shape[1]
    elapsed_min = (time.time() - t0) / 60.0
    return {
        "batch": batch_label,
        "prediction": res.prediction,
        "cna_mat": res.cna_mat,
        "subclone": res.subclone,
        "warnings": res.warnings,
        "elapsed_min": elapsed_min,
        "n_cells_in": n_cells_in,
        "n_cells_out": int(res.prediction.shape[0]),
    }


def copykat_by_batch(
    adata: "AnnData",
    batch_key: str,
    *,
    layer: str | None = None,
    config: CopykatConfig | None = None,
    min_cells_per_batch: int = 100,
    n_batch_parallel: int = 1,
    on_error: str = "skip",
    cna_storage: str = "obsm",
    cna_dtype: str = "float32",
    copy: bool = True,
) -> "AnnData":
    """Run copykat per batch on an AnnData and merge results back in.

    Parameters
    ----------
    adata
        cells × genes AnnData. ``adata.X`` (or ``layer``) must be raw counts.
    batch_key
        Column in ``adata.obs`` that partitions cells into batches.
    layer
        Optional layer name with raw counts; if ``None``, uses ``adata.X``.
    config
        Baseline :class:`CopykatConfig`; ``sam_name`` will be suffixed by the
        batch label for each run. ``output_dir`` is shared across batches.
    min_cells_per_batch
        Batches with fewer cells than this are skipped (logged in ``uns``).
    n_batch_parallel
        Worker processes to run different batches concurrently. Total CPU
        usage is ``n_batch_parallel × config.n_jobs``; the default ``1`` keeps
        runs sequential to avoid CPU contention with the per-batch thread pool.
    on_error
        ``"skip"`` logs the traceback in ``uns`` and continues with remaining
        batches; ``"raise"`` re-raises the first exception.
    cna_storage
        ``"obsm"`` stores the per-cell × per-bin CNA matrix in
        ``adata.obsm[copykat_cna]`` (rows for skipped/filtered cells are NaN);
        ``"none"`` omits it (only predictions are attached).
    cna_dtype
        Dtype for the obsm CNA matrix; ``"float32"`` halves memory vs
        ``"float64"`` without losing signal for log-ratio-scale values.
    copy
        If ``True`` (default), operates on ``adata.copy()`` and returns the
        modified copy; if ``False``, mutates ``adata`` in place and returns it.

    Returns
    -------
    AnnData
        With the following additions:

        - ``obs[copykat_pred]``       — "diploid" / "aneuploid" / "not.defined"
        - ``obs[copykat_subclone]``   — integer subclone label (batch-local)
        - ``obs[copykat_batch]``      — batch label actually used
        - ``obs[copykat_warnings]``   — comma-separated warnings for that batch
        - ``obsm[copykat_cna]``       — float32 (n_cells, n_bins) CNA matrix
        - ``uns[copykat_bins]``       — DataFrame of bin coordinates
        - ``uns[copykat_per_batch]``  — dict of per-batch metadata (timings,
          cell counts, warnings, status)
    """
    if config is None:
        config = CopykatConfig()
    if batch_key not in adata.obs.columns:
        raise KeyError(
            f"batch_key={batch_key!r} not in adata.obs; "
            f"available: {list(adata.obs.columns)[:20]}"
        )
    if cna_storage not in ("obsm", "none"):
        raise ValueError(f"cna_storage must be 'obsm' or 'none', got {cna_storage!r}")
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")

    ad = adata.copy() if copy else adata
    obs_names = ad.obs_names.astype(str)
    batch_series = ad.obs[batch_key].astype(str)

    # Pre-allocate CNA matrix using the shared hg20 bin table; every batch
    # returns the same (n_bins, n_cells) shape so we can fold batches into
    # one obsm array without alignment pain.
    bins_df = load_hg20_bins()
    bins_retained = bins_df.loc[bins_df["chrom"] != 24].reset_index(drop=True).copy()
    bins_retained["end"] = bins_retained["chrompos"]
    bins_retained["start"] = (
        bins_retained.groupby("chrom", sort=False)["end"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float64)
    )
    n_bins = int(len(bins_retained))
    n_cells = int(ad.n_obs)

    if cna_storage == "obsm":
        cna_out = np.full((n_cells, n_bins), np.nan, dtype=np.dtype(cna_dtype))
    else:
        cna_out = None

    pred_out = np.array(["not.defined"] * n_cells, dtype=object)
    subclone_out = np.full(n_cells, np.nan, dtype=np.float64)
    warn_out = np.array([""] * n_cells, dtype=object)
    batch_out = batch_series.to_numpy().astype(object)

    per_batch_meta: dict[str, dict[str, object]] = {}
    to_run: list[tuple[str, "pd.DataFrame | _SparseBatch"]] = []

    for batch_label, idx in batch_series.groupby(batch_series).groups.items():
        cell_mask = np.asarray(idx.to_numpy(), dtype=object)
        n_in = len(cell_mask)
        if n_in < min_cells_per_batch:
            per_batch_meta[str(batch_label)] = {
                "status": "skipped_too_small",
                "n_cells_in": n_in,
                "reason": f"{n_in} < min_cells_per_batch ({min_cells_per_batch})",
            }
            continue
        counts = _extract_counts_frame(ad, cell_mask, layer)
        to_run.append((str(batch_label), counts))

    def _wrap(
        pair: tuple[str, "pd.DataFrame | _SparseBatch"],
    ) -> tuple[str, dict[str, object] | BaseException]:
        label, cnt = pair
        try:
            return label, _run_one_batch(label, cnt, config)
        except BaseException as exc:  # noqa: BLE001 - funnel exc to caller
            return label, exc

    # Sequential by default; cross-batch parallel opt-in. When no batches meet
    # the size threshold we still fall through and attach empty obs/obsm so
    # downstream code can assume the columns exist.
    if not to_run:
        results: list[tuple[str, dict[str, object] | BaseException]] = []
    elif n_batch_parallel <= 1:
        results = [_wrap(p) for p in to_run]
    else:
        results = Parallel(n_jobs=n_batch_parallel, prefer="processes")(
            delayed(_wrap)(p) for p in to_run
        )

    for label, payload in results:
        if isinstance(payload, BaseException):
            if on_error == "raise":
                raise payload
            per_batch_meta[label] = {
                "status": "error",
                "error": f"{type(payload).__name__}: {payload}",
            }
            continue
        pb = payload  # dict
        pred_df = pb["prediction"]  # type: ignore[assignment]
        cna_df = pb["cna_mat"]  # type: ignore[assignment]
        subclone_s = pb["subclone"]  # type: ignore[assignment]
        warnings = pb["warnings"]  # type: ignore[assignment]

        # Map per-cell results back into the parent-sized arrays
        pred_map = dict(zip(pred_df["cell"].astype(str), pred_df["copykat.pred"].astype(str)))
        for i, name in enumerate(obs_names):
            if name in pred_map:
                pred_out[i] = pred_map[name]
                warn_out[i] = ", ".join(warnings) if warnings else ""

        if len(subclone_s) > 0:
            sc_map = {str(k): int(v) for k, v in subclone_s.items()}
            for i, name in enumerate(obs_names):
                if name in sc_map:
                    subclone_out[i] = sc_map[name]

        if cna_out is not None:
            # cna_df is a bins × cells DataFrame; align columns to obs_names
            col_pos = {c: j for j, c in enumerate(cna_df.columns.astype(str))}
            for i, name in enumerate(obs_names):
                j = col_pos.get(name)
                if j is not None:
                    cna_out[i, :] = cna_df.values[:, j].astype(cna_dtype, copy=False)

        per_batch_meta[label] = {
            "status": "ok",
            "elapsed_min": float(pb["elapsed_min"]),
            "n_cells_in": int(pb["n_cells_in"]),
            "n_cells_out": int(pb["n_cells_out"]),
            "warnings": list(warnings),
            "subclone_n": int(len(subclone_s)),
        }

    ad.obs[_PRED_COL] = pd.Categorical(
        pred_out,
        categories=sorted(set(pred_out.tolist()) | {"not.defined"}),
    )
    ad.obs[_SUBCLONE_COL] = subclone_out
    ad.obs[_BATCH_COL] = pd.Categorical(batch_out)
    ad.obs[_WARN_COL] = warn_out

    if cna_out is not None:
        ad.obsm[_CNA_KEY] = cna_out
        ad.uns[_BINS_KEY] = bins_retained[["chrom", "start", "end"]].reset_index(drop=True)

    ad.uns[_META_KEY] = per_batch_meta
    return ad
