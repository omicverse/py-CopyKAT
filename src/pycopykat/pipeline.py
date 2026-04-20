"""Top-level :func:`copykat` pipeline orchestration (R copykat 10-step flow).

Strictly follows the R source ordering (``copykat-R/R/copykat.R``):

1. LOW.DR + min_gene_per_cell filter (`filter_cells_and_genes`)
2. Annotate gene coordinates (`annotate_genes`, HLA + cell-cycle removed)
3. Stage-1 per-chromosome coverage filter (`filter_cells_by_chrom_coverage`)
4. VST (log(sqrt(x)+sqrt(x+1))) + per-cell centering (`vst_center`)
5. Kalman smoothing + re-center (`smooth_cells`)
6. Baseline estimation (`baseline_norm_cl`, fallback `baseline_gmm`,
   or `baseline_synthetic` for cell-line)
7. UP.DR filter on raw counts + stage-2 chrom-coverage filter
8. Per-cell Poisson-Gamma segmentation with cluster-union breaks (`segment_cells`)
9. Gene → 220 kb bin aggregation (`aggregate_to_bins`)
10. Diploid-anchored baseline adjustment (`baseline_adjust`)
11. Diploid/aneuploid prediction (`predict_ploidy`)
12. Subclone detection (`dynamic_tree_cut`) on aneuploid cells

Returns a :class:`CopykatResult`.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage

from pycopykat.baseline.auto import baseline_norm_cl
from pycopykat.baseline.gmm import baseline_gmm
from pycopykat.baseline.synthetic import baseline_synthetic
from pycopykat.classify.adjust_pipeline import baseline_adjust
from pycopykat.classify.predict import predict_ploidy
from pycopykat.classify.subclone import dynamic_tree_cut
from pycopykat.cna.bins import aggregate_to_bins
from pycopykat.config import CopykatConfig
from pycopykat.io.annotation import annotate_genes, load_hg20_bins
from pycopykat.kernels.distances import (
    pdist_euclidean,
    pdist_pearson,
    pdist_spearman,
)
from pycopykat.preprocess.filter import (
    filter_cells_and_genes,
    filter_cells_by_chrom_coverage,
)
from pycopykat.preprocess.normalize import vst_center
from pycopykat.preprocess.smooth import smooth_cells
from pycopykat.result import CopykatResult
from pycopykat.segment.mcmc import segment_cells

log = logging.getLogger("pycopykat.pipeline")

_ANNO_COLS = (
    "hgnc_symbol",
    "ensembl_gene_id",
    "chromosome_name",
    "start_position",
    "end_position",
    "abspos",
    "band",
)

_DIST_FN: dict[str, Callable[[NDArray[np.floating]], NDArray[np.float64]]] = {
    "euclidean": pdist_euclidean,
    "pearson": pdist_pearson,
    "spearman": pdist_spearman,
}


def _subset_gene_rows(
    gene_anno: pd.DataFrame,
    raw_counts: NDArray[np.float64],
    keep_mask: NDArray[np.bool_],
) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    return gene_anno.loc[keep_mask].reset_index(drop=True), raw_counts[keep_mask]


def _cluster_for_segmentation(
    norm_relat: NDArray[np.float64], distance: str, n_cells: int
) -> NDArray[np.int_]:
    """Ward-cluster the relative-expression cells for segmentation consensus."""
    d = _DIST_FN[distance](norm_relat.T)
    Z = linkage(d, method="ward")
    k = min(6, max(2, n_cells // 5))
    return fcluster(Z, t=k, criterion="maxclust").astype(np.int_)


def copykat(
    mat: pd.DataFrame,
    config: CopykatConfig | None = None,
    **kwargs: object,
) -> CopykatResult:
    """Full CopyKAT pipeline.

    Parameters
    ----------
    mat
        Genes × cells raw count DataFrame. Index holds gene identifiers.
    config
        Optional :class:`CopykatConfig`. Overrides defaults. If omitted,
        ``kwargs`` populate the config.
    """
    cfg = config or CopykatConfig(**kwargs)  # type: ignore[arg-type]
    if not isinstance(mat, pd.DataFrame):
        raise TypeError("pass a pandas DataFrame with gene names in the index")

    warnings: list[str] = []

    # ── step 1: filter cells + genes by LOW.DR ────────────────────────────────
    log.info("step 1/12: filter cells & genes")
    mat1, stat1 = filter_cells_and_genes(
        mat, min_gene_per_cell=cfg.min_gene_per_cell, low_dr=cfg.low_dr
    )
    if stat1["data_quality"] == "low":
        warnings.append("low data quality — UP.DR set to LOW.DR")
        up_dr = cfg.low_dr
    else:
        up_dr = cfg.up_dr

    # ── step 2: annotate ─────────────────────────────────────────────────────
    log.info("step 2/12: annotate gene coordinates")
    annotated = annotate_genes(mat1, id_type=cfg.id_type, genome=cfg.genome)
    cell_cols = [c for c in annotated.columns if c not in _ANNO_COLS]
    anno_present = [c for c in _ANNO_COLS if c in annotated.columns]
    gene_anno = annotated[anno_present].copy().reset_index(drop=True)
    raw_counts = annotated[cell_cols].to_numpy(dtype=np.float64)

    # ── step 3: stage-1 chromosome-coverage cell filter ───────────────────────
    log.info("step 3/12: stage-1 chromosome coverage filter")
    chrom_int = gene_anno["chromosome_name"].to_numpy(dtype=np.int64)
    _, dropped1 = filter_cells_by_chrom_coverage(
        raw_counts, chrom_int, ngene_chr=cfg.ngene_chr
    )
    keep1 = np.ones(len(cell_cols), dtype=bool)
    keep1[dropped1] = False
    cell_cols = [c for i, c in enumerate(cell_cols) if keep1[i]]
    raw_counts = raw_counts[:, keep1]
    if len(cell_cols) < 10:
        raise ValueError(
            f"too few cells after stage-1 chrom coverage filter: {len(cell_cols)}"
        )

    # ── step 4: VST + center ─────────────────────────────────────────────────
    log.info("step 4/12: VST + per-cell centering")
    X_norm = vst_center(raw_counts)

    # ── step 5: Kalman smoothing ─────────────────────────────────────────────
    log.info("step 5/12: Kalman smoothing")
    X_smooth = smooth_cells(X_norm)

    # ── step 6: baseline estimation ───────────────────────────────────────────
    log.info("step 6/12: baseline estimation")
    # `auto_labels` holds the Ward cluster labels produced by `baseline_norm_cl`
    # on the **pre-stage-2** cells, to be reused for segmentation clustering in
    # step 8 (A3). Only populated on the happy auto-baseline path — on any
    # other branch (cell-line, norm_cell_names, GMM fallback) we fall back to
    # a fresh `_cluster_for_segmentation` call.
    auto_labels: NDArray[np.int_] | None = None
    if cfg.cell_line:
        # V1: simplified cell-line mode — global median across all cells.
        # Proper per-cluster synthetic-normal (baseline_synthetic) requires
        # realignment of column ordering and is deferred to V2.
        basel = np.median(X_smooth, axis=1)
        preN: list[str] = []
        warnings.append("cell-line mode (V1 simplified: global median baseline)")
    elif cfg.norm_cell_names:
        known_set = set(cfg.norm_cell_names)
        known_mask = np.array([c in known_set for c in cell_cols])
        if int(known_mask.sum()) < 2:
            raise ValueError("norm_cell_names requires at least 2 matches in cells")
        basel = np.median(X_smooth[:, known_mask], axis=1)
        preN = [c for c, m in zip(cell_cols, known_mask) if m]
    else:
        br = baseline_norm_cl(
            X_smooth, cell_names=cell_cols, min_cells=5, seed=cfg.seed
        )
        if br.warning:
            warnings.append(br.warning)
            log.info("auto baseline flagged unclassified → fallback to GMM")
            br = baseline_gmm(
                X_smooth, cell_names=cell_cols, seed=cfg.seed, fallback=br
            )
            if br.warning:
                warnings.append(br.warning)
        else:
            # Only the happy `baseline_norm_cl` path gives labels that match the
            # segmentation stage's needs; GMM labels are produced on a different
            # (post-fallback) tree and the synthetic/cell-line branches have no
            # labels at all.
            auto_labels = np.asarray(br.labels, dtype=np.int_)
        basel = br.basel
        preN = br.preN
    norm_relat = X_smooth - basel[:, None]

    # ── step 7: UP.DR filter (on raw counts) + stage-2 chrom coverage ─────────
    log.info("step 7/12: UP.DR + stage-2 chromosome coverage filter")
    DR2 = (raw_counts > 0).sum(axis=1) / raw_counts.shape[1]
    keep_g = DR2 >= up_dr
    gene_anno, raw_counts = _subset_gene_rows(gene_anno, raw_counts, keep_g)
    norm_relat = norm_relat[keep_g]
    chrom_int = gene_anno["chromosome_name"].to_numpy(dtype=np.int64)

    _, dropped2 = filter_cells_by_chrom_coverage(
        raw_counts, chrom_int, ngene_chr=cfg.ngene_chr
    )
    keep2 = np.ones(len(cell_cols), dtype=bool)
    keep2[dropped2] = False
    cell_cols = [c for i, c in enumerate(cell_cols) if keep2[i]]
    raw_counts = raw_counts[:, keep2]
    norm_relat = norm_relat[:, keep2]
    preN = [c for c in preN if c in set(cell_cols)]
    if len(cell_cols) < 10:
        raise ValueError(
            f"too few cells after stage-2 chrom coverage filter: {len(cell_cols)}"
        )

    # ── step 8: per-cell segmentation ────────────────────────────────────────
    log.info("step 8/12: per-cell Poisson-Gamma segmentation")
    # A3: reuse the Ward labels from baseline_norm_cl when available. Those
    # labels were computed BEFORE stage-2 filtering, so subset by keep2 and
    # re-label to consecutive ids. Fall back to a fresh ward cut if fewer
    # than two distinct labels survive, or if auto_labels is not available.
    clu: NDArray[np.int_] | None = None
    if auto_labels is not None:
        sub = auto_labels[keep2]
        uniq = np.unique(sub)
        if uniq.size >= 2:
            _, inv = np.unique(sub, return_inverse=True)
            clu = (inv + 1).astype(np.int_)
    if clu is None:
        clu = _cluster_for_segmentation(norm_relat, cfg.distance, len(cell_cols))
    logCNA, _BR = segment_cells(
        norm_relat,
        clu,
        bins=cfg.win_size,
        ks_cut=cfg.ks_cut,
        seed=cfg.seed,
        mc=1000,
        n_jobs=cfg.n_jobs,
    )

    # ── step 9: aggregate to 220 kb bins ─────────────────────────────────────
    log.info("step 9/12: aggregate genes to 220 kb bins")
    bins_df = load_hg20_bins()
    cna_bins = aggregate_to_bins(logCNA, gene_anno, bins_df, exclude_chrom_24=True)
    # bins used in cna_bins: same rows as bins_df minus chrom 24
    bins_retained = bins_df.loc[bins_df["chrom"] != 24].reset_index(drop=True).copy()
    bins_retained["end"] = bins_retained["chrompos"]
    bins_retained["start"] = (
        bins_retained.groupby("chrom", sort=False)["end"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float64)
    )

    # ── step 10: baseline-anchored adjustment ────────────────────────────────
    log.info("step 10/12: baseline-anchored CNA adjustment")
    preN_set = set(preN) if preN else set()
    diploid_mask = np.array([c in preN_set for c in cell_cols])
    if int(diploid_mask.sum()) < 2:
        # fallback: pseudo-diploid from the first few cells
        diploid_mask[: max(2, min(5, len(cell_cols)))] = True
        warnings.append("diploid_mask fallback (<2 preN cells)")
    cna_adj = baseline_adjust(cna_bins, diploid_mask, factor=0.25)

    # ── step 11: predict diploid / aneuploid ──────────────────────────────────
    log.info("step 11/12: diploid/aneuploid prediction")
    pred_preN = list(preN_set) if preN_set else [
        c for c, m in zip(cell_cols, diploid_mask) if m
    ]
    pred = predict_ploidy(
        cna_adj, cell_cols, preN=pred_preN, distance=cfg.distance
    )

    # Tag low-confidence predictions per R behaviour
    if any("unclassified.prediction" in w for w in warnings):
        pred = pred.copy()
        pred["copykat.pred"] = pred["copykat.pred"].replace({
            "diploid": "c1:diploid:low.conf",
            "aneuploid": "c2:aneuploid:low.conf",
        })

    # ── step 12: subclone detection on aneuploid cells ───────────────────────
    log.info("step 12/12: subclone detection")
    aneu_mask_s = pred["copykat.pred"].astype(str).str.contains("aneuploid")
    aneu_cells = pred.loc[aneu_mask_s, "cell"].tolist()
    if len(aneu_cells) >= 20:
        aneu_idx = [i for i, c in enumerate(cell_cols) if c in set(aneu_cells)]
        cna_aneu = cna_adj[:, aneu_idx]
        Zs = linkage(_DIST_FN[cfg.distance](cna_aneu.T), method="ward")
        min_sz = max(10, len(aneu_cells) // 20)
        sub_labels = dynamic_tree_cut(Zs, min_cluster_size=min_sz, deep_split=2)
        subclone = pd.Series(
            sub_labels.astype(int),
            index=[cell_cols[i] for i in aneu_idx],
            name="subclone",
        )
    else:
        subclone = pd.Series([], dtype=int, name="subclone")

    # ── assemble result ──────────────────────────────────────────────────────
    cna_df = pd.DataFrame(
        cna_adj,
        index=pd.MultiIndex.from_frame(bins_retained[["chrom", "start", "end"]]),
        columns=cell_cols,
    )
    Z_final = linkage(_DIST_FN[cfg.distance](cna_adj.T), method="ward")

    return CopykatResult(
        cna_mat=cna_df,
        prediction=pred.reset_index(drop=True),
        linkage=Z_final,
        subclone=subclone,
        warnings=tuple(w for w in warnings if w),
    )
