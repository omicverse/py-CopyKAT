"""Public entry point :func:`copykat` (R ``copykat()``)."""
from __future__ import annotations

import logging
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy import sparse

from pycopykat import core
from pycopykat.config import CopykatConfig
from pycopykat.result import CopykatResult

log = logging.getLogger("pycopykat.pipeline")


@contextmanager
def _threads(n: int):
    """Limit BLAS and numba to ``n`` threads."""
    import numba
    from threadpoolctl import threadpool_limits

    old = numba.get_num_threads()
    numba.set_num_threads(max(1, min(n, numba.config.NUMBA_NUM_THREADS)))
    try:
        with threadpool_limits(limits=n):
            yield
    finally:
        numba.set_num_threads(old)


def _run(X, genes, cells, cfg: CopykatConfig) -> CopykatResult:
    with _threads(cfg.n_jobs):
        run = core.run(
            X, genes, cells, id_type="E" if cfg.id_type == "Ensembl" else "S",
            cell_line=cfg.cell_line, ngene_chr=cfg.ngene_chr, min_gene_per_cell=cfg.min_gene_per_cell,
            LOW_DR=cfg.low_dr, UP_DR=cfg.up_dr, win_size=cfg.win_size,
            norm_cell_names=cfg.norm_cell_names, KS_cut=cfg.ks_cut, distance=cfg.distance,
            genome=cfg.genome, seed=cfg.seed, n_jobs=cfg.n_jobs)
    npos = 3 if cfg.genome == "hg20" else 7
    pos = run.cna.iloc[:, :npos]
    cna = run.cna.iloc[:, npos:]
    cna.index = pd.MultiIndex.from_frame(pos)
    pred = run.prediction if run.prediction is not None else pd.DataFrame(
        {"cell.names": pd.Series(dtype=str), "copykat.pred": pd.Series(dtype=str)})
    warnings = tuple(w for w in (run.WNS1 if run.WNS1 != "data quality is ok" else "", run.WNS) if w)
    return CopykatResult(prediction=pred.reset_index(drop=True), cna_mat=cna,
                         hclustering=run.hclustering, warnings=warnings, preN=list(run.preN))


def copykat(mat, config: CopykatConfig | None = None, **kwargs: object) -> CopykatResult:
    """Run CopyKAT (R copykat 1.2.5) on raw UMI counts.

    Parameters
    ----------
    mat
        Genes x cells counts: a :class:`pandas.DataFrame` with gene names in the index and cell
        names in the columns, or an :class:`anndata.AnnData` (cells x genes; ``adata.X`` holds
        the counts).
    config
        A :class:`CopykatConfig`; keyword arguments fill one when it is omitted.
    """
    cfg = config or CopykatConfig(**kwargs)  # type: ignore[arg-type]
    if isinstance(mat, pd.DataFrame):
        X = mat.to_numpy(dtype=np.float64)
        return _run(X, mat.index.astype(str).to_numpy(), mat.columns.astype(str).to_numpy(), cfg)
    if hasattr(mat, "obs_names") and hasattr(mat, "var_names"):
        X = mat.X
        X = X.T.tocsc() if sparse.issparse(X) else np.asarray(X, dtype=np.float64).T
        return _run(X, mat.var_names.astype(str).to_numpy(), mat.obs_names.astype(str).to_numpy(), cfg)
    raise TypeError("pass a genes x cells pandas DataFrame or an AnnData")


def _copykat_from_sparse(X_sparse, gene_names, cell_names, config: CopykatConfig) -> CopykatResult:
    """Genes x cells sparse counts with their names."""
    return _run(sparse.csc_matrix(X_sparse), np.asarray(gene_names).astype(str),
                np.asarray(cell_names).astype(str), config)
