"""The CopyKAT algorithm, following R copykat 1.2.5 (``copykat.R`` and its helpers) step by step.

With ``n.cores = 1`` R copykat is reproducible: it calls ``set.seed(1234)`` and then draws only in
the segmentation (MCMCpack's ``MCpoissongamma``, i.e. ``rgamma``) and, in cell-line mode, in
``baseline.synthetic`` (``set.seed(123)`` and ``rnorm``). This module draws the same numbers from
the same stream (:mod:`pycopykat._rcompat`), uses R's ``hclust``/``cutree`` and mixtools' EM, and
keeps R's order of genes and cells, so the same seed gives the same result as in R.

(With ``n.cores > 1``, R's ``mclapply`` children start with unseeded generators, so R copykat
itself does not repeat its own results; this port always follows the ``n.cores = 1`` stream and
uses threads only for steps without random numbers.)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import f as f_dist

from pycopykat._rcompat.hclust import HClust, hclust
from pycopykat._rcompat.mixtools import normalmix3
from pycopykat._rcompat.rbase import r_means, r_medians, r_sds
from pycopykat._rcompat.rgamma import GammaStream
from pycopykat._rcompat.rng import RRandom
from pycopykat.io.annotation import load_genome_annotation, load_hg20_bins, load_hg20_cycle_genes
from pycopykat.kernels.kalman import kalman_smooth_matrix

log = logging.getLogger("pycopykat")

ANNO_COLS = ("abspos", "chromosome_name", "start_position", "end_position", "ensembl_gene_id",
             "hgnc_symbol", "band")


@dataclass
class CopykatRun:
    """Everything R copykat returns, plus the intermediate objects used to compare with R."""

    prediction: pd.DataFrame | None          # cell.names, copykat.pred (R order)
    cna: pd.DataFrame                        # CNAmat: position columns + one column per cell
    hclustering: HClust
    cells: list[str]
    WNS1: str = ""
    WNS: str = ""
    preN: list[str] = field(default_factory=list)
    intermediate: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------
# helpers


def euclidean_dist(X: np.ndarray) -> np.ndarray:
    """Euclidean distances between the columns of ``X`` (``parDist(t(X))``). Rows are centred
    first, which leaves the distances unchanged and keeps the Gram-matrix form accurate."""
    Xc = X - X.mean(axis=1, keepdims=True)
    sq = np.einsum("ij,ij->j", Xc, Xc)
    g = Xc.T @ Xc
    d2 = sq[:, None] + sq[None, :] - 2.0 * g
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2)


def cor_dist(X: np.ndarray, method: str) -> np.ndarray:
    """``as.dist(1 - cor(X, method = method))`` between columns."""
    if method == "spearman":
        X = np.apply_along_axis(lambda v: pd.Series(v).rank().to_numpy(), 0, X)
    return 1.0 - np.corrcoef(X, rowvar=False)


def _cluster_dist(X: np.ndarray, distance: str) -> np.ndarray:
    return euclidean_dist(X) if distance == "euclidean" else cor_dist(X, distance)


def _cut_min_size(tree: HClust, start: int, min_cells: int) -> np.ndarray:
    """``km <- start; ct <- cutree(fit, km); while(!all(table(ct) > min)) {km <- km-1; ct <-
    cutree(fit, km); if (km == 2) break}``."""
    km = start
    ct = tree.cutree(km)
    while not np.all(np.bincount(ct)[1:] > min_cells):
        km -= 1
        ct = tree.cutree(km)
        if km == 2:
            break
    return ct


def _final_k(tree: HClust) -> int:
    """The v1.2.0 rule: the largest k in 4, 3, 2 whose smallest cluster has more than 10 cells."""
    for k in (4, 3, 2):
        if np.bincount(tree.cutree(k))[1:].min() > 10:
            return k
    raise ValueError("no cut of the cell tree into 2 to 4 clusters has more than 10 cells in each "
                     "cluster (R copykat stops here)")


def silhouette_mean(d: np.ndarray, cl: np.ndarray) -> float:
    """``mean(cluster::silhouette(cl, d)[, "sil_width"])``."""
    labs = np.unique(cl)
    n = cl.size
    sums = np.column_stack([d[:, cl == k].sum(axis=1) for k in labs])
    sizes = np.array([(cl == k).sum() for k in labs])
    own = np.searchsorted(labs, cl)
    a_cnt = sizes[own] - 1
    a = np.where(a_cnt > 0, sums[np.arange(n), own] / np.maximum(a_cnt, 1), 0.0)
    other = sums / sizes[None, :]
    other[np.arange(n), own] = np.inf
    b = other.min(axis=1)
    s = np.where(a_cnt > 0, (b - a) / np.maximum(a, b), 0.0)
    return float(np.mean(s))


def _chrom_runs_filter(nz_rows: list[np.ndarray] | None, X: sparse.csc_matrix, chrom: np.ndarray,
                       n_chrom: int, ngene_chr: int) -> np.ndarray:
    """R's per-cell chromosome-coverage test (``ToRemov2``/``ToRemov3``): a cell is removed when
    it has fewer than 3 expressed genes, or the runs of chromosomes over its expressed genes (in
    genomic order) are fewer than the chromosomes, or the shortest run is below ``ngene_chr``."""
    X = X.tocsc()
    X.sort_indices()
    X.eliminate_zeros()
    nnz = np.diff(X.indptr)
    ch = chrom[X.indices]
    cell = np.repeat(np.arange(X.shape[1]), nnz)
    new_run = np.ones(ch.size, dtype=bool)
    new_run[1:] = (ch[1:] != ch[:-1]) | (cell[1:] != cell[:-1])
    run_id = np.cumsum(new_run) - 1
    run_len = np.bincount(run_id)
    run_cell = cell[new_run]
    n_runs = np.bincount(run_cell, minlength=X.shape[1])
    min_run = np.full(X.shape[1], np.iinfo(np.int64).max)
    np.minimum.at(min_run, run_cell, run_len)
    return (nnz < 3) | (n_runs < n_chrom) | (min_run < ngene_chr)


# ------------------------------------------------------------------------------------------------
# baselines


def baseline_norm_cl(smooth: np.ndarray, cells: list[str], min_cells: int = 10):
    """R ``baseline.norm.cl``."""
    d = euclidean_dist(smooth)
    fit = hclust(d, "ward.D2")
    ct = _cut_min_size(fit, 6, min_cells)
    k = int(ct.max())
    data = np.empty((k, smooth.shape[0]))
    for i in range(1, k + 1):
        cols = np.flatnonzero(ct == i)
        if cols.size < 2:
            raise ValueError("a cluster of the baseline step has a single cell (R copykat stops here)")
        data[i - 1] = r_medians(smooth[:, cols], axis=1)
    sx = np.maximum(0.05, 0.5 * r_sds(data, axis=1))
    gm = normalmix3(data, sx, maxit=5000)
    SDM = gm.sigma
    wn = silhouette_mean(d, fit.cutree(2))
    n = smooth.shape[0]
    PDt = float(f_dist.sf(SDM.max() ** 2 / SDM.min() ** 2, n, n))
    ok_size = bool(np.all(np.bincount(ct)[1:] > min_cells))
    WNS = "unclassified.prediction" if (wn <= 0.15 or not ok_size or PDt > 0.05) else ""
    low = np.flatnonzero(SDM == SDM.min()) + 1
    keep = np.isin(ct, low)
    basel = r_medians(smooth[:, keep], axis=1)
    preN = [c for c, k_ in zip(cells, keep) if k_]
    info = dict(SDM=SDM, silhouette=wn, PDt=PDt, em_iterations=gm.iterations)
    return basel, WNS, preN, ct, info


def baseline_gmm(smooth: np.ndarray, cells: list[str], max_normal: int = 5, mu_cut: float = 0.05,
                 nfraq_cut: float = 0.99, chunk: int = 64):
    """R ``baseline.GMM``: per-cell three-component EM, cells in order until ``max_normal`` are
    called diploid. Returns ``(basel, preN)`` or ``None`` when two or fewer are found."""
    n_cells = smooth.shape[1]
    found: list[int] = []
    for c0 in range(0, n_cells, chunk):
        cols = np.arange(c0, min(n_cells, c0 + chunk))
        sam = smooth[:, cols].T
        sg = np.maximum(0.05, 0.5 * r_sds(sam, axis=1))
        gm = normalmix3(sam, sg, maxit=500)
        near = np.abs(gm.mu) <= mu_cut
        frq = (gm.lam * near).sum(axis=1)
        dip = near.any(axis=1) & (frq > nfraq_cut)
        for j in np.flatnonzero(dip):
            found.append(int(cols[j]))
            if len(found) >= max_normal:
                break
        if len(found) >= max_normal:
            break
    if len(found) <= 2:
        return None
    basel = r_means(smooth[:, found], axis=1)
    return basel, [cells[j] for j in found]


def baseline_synthetic(smooth: np.ndarray, cells: list[str], rng: RRandom, min_cells: int = 10):
    """R ``baseline.synthetic``: per cluster, subtract one ``rnorm`` per gene with the cluster's
    per-gene sd, after ``set.seed(123)`` in every cluster. Columns come out grouped by cluster."""
    d = euclidean_dist(smooth)
    fit = hclust(d, "ward.D2")
    ct = _cut_min_size(fit, 6, min_cells)
    parts, order = [], []
    for i in range(1, int(ct.max()) + 1):
        cols = np.flatnonzero(ct == i)
        sd1 = r_sds(smooth[:, cols], axis=1)
        rng.set_seed(123)
        syn = rng.rnorm(np.zeros(sd1.size), sd1)
        parts.append(smooth[:, cols] - syn[:, None])
        order.extend(cols.tolist())
    relat = np.concatenate(parts, axis=1)
    return relat, [cells[j] for j in order], ct


# ------------------------------------------------------------------------------------------------
# segmentation


def cna_mcmc(clu: np.ndarray, fttmat: np.ndarray, bins: int, cut_cor: float, gamma: GammaStream,
             mc: int = 1000):
    """R ``CNA.MCMC``: breakpoints from the cluster medians, then a Poisson-gamma posterior mean
    for every cell on every segment. Returns ``(logCNA, BR)`` (BR 1-based, as R)."""
    n = fttmat.shape[0]
    lo, hi = int(clu.min()), int(clu.max())
    CON = np.empty((n, hi - lo + 1))
    for j, i in enumerate(range(lo, hi + 1)):
        cols = np.flatnonzero(clu == i)
        if cols.size < 2:
            raise ValueError("a segmentation cluster has fewer than two cells (R copykat stops here)")
        CON[:, j] = r_medians(fttmat[:, cols], axis=1)
    sm = np.exp(CON)
    breks = np.concatenate([np.arange(1, int(n / bins - 1) * bins + 1, bins), [n]]).astype(np.int64)
    m = breks.size - 2
    # windows i: breks[i]..breks[i+1] and breks[i+1]+1..breks[i+2] (1-based, inclusive)
    s1, l1 = breks[:m] - 1, breks[1:m + 1] - breks[:m] + 1
    s2, l2 = breks[1:m + 1], breks[2:m + 2] - breks[1:m + 1]
    BR: set[int] = set()
    for c in range(sm.shape[1]):
        sum1, mean1 = _window_stats(sm[:, c], s1, l1)
        sum2, mean2 = _window_stats(sm[:, c], s2, l2)
        shapes = np.empty(2 * m)
        rates = np.empty(2 * m)
        shapes[0::2] = np.maximum(mean1, 0.001) + sum1
        shapes[1::2] = np.maximum(mean2, 0.001) + sum2
        rates[0::2] = 1.0 + l1
        rates[1::2] = 1.0 + l2
        draws = gamma.calls(shapes, rates, mc, draws=True)
        bre = [int(breks[i + 1]) for i in range(m) if ks_statistic(draws[2 * i], draws[2 * i + 1]) > cut_cor]
        BR.update(sorted(set([1, *bre, n])))
    BR_arr = np.array(sorted(BR), dtype=np.int64)
    logCNA = _segment_cells(np.exp(fttmat), BR_arr, gamma, mc)
    return logCNA, BR_arr


def _segment_cells(sm: np.ndarray, BR: np.ndarray, gamma: GammaStream, mc: int) -> np.ndarray:
    """``seg(z)`` for every cell: segment i covers ``BR[i]..BR[i+1]`` (both ends), later segments
    overwrite the shared end; ``x <- log(x)``."""
    n, n_cells = sm.shape
    nseg = BR.size - 1
    starts = BR[:-1] - 1
    ends = BR[1:]                                         # exclusive, 0-based
    lens = (ends - starts).astype(np.float64)
    # per cell and segment: R mean and long-double sum of the segment values
    sums = np.empty((n_cells, nseg))
    means = np.empty((n_cells, nseg))
    for s_ in range(nseg):
        block = sm[starts[s_]:ends[s_], :]
        sums[:, s_] = _ld_sum_cols(block)
        means[:, s_] = r_means(block, axis=0)
    a = np.maximum(means, 0.001)
    shapes = (a + sums).ravel()                           # cell-major: cell 1 segments, cell 2 ...
    rates = np.tile(1.0 + lens, n_cells)
    post = gamma.calls(shapes, rates, mc).reshape(n_cells, nseg)
    x = np.zeros((n, n_cells))
    for s_ in range(nseg):
        x[starts[s_]:ends[s_], :] = post[:, s_][None, :]
    with np.errstate(divide="ignore"):
        return np.log(x)


def _window_stats(v: np.ndarray, starts: np.ndarray, lens: np.ndarray):
    """R ``sum`` and ``mean`` of the windows ``v[starts[i]:starts[i] + lens[i]]``, added in
    order in long double, for all windows at once."""
    ld = np.longdouble
    s = np.zeros(starts.size, dtype=ld)
    for j in range(int(lens.max())):
        on = j < lens
        s[on] += v[starts[on] + j]
    mu = s / lens
    t = np.zeros(starts.size, dtype=ld)
    for j in range(int(lens.max())):
        on = j < lens
        t[on] += v[starts[on] + j].astype(ld) - mu[on]
    return s.astype(np.float64), (mu + t / lens).astype(np.float64)


def _ld_sum_cols(block: np.ndarray) -> np.ndarray:
    s = np.zeros(block.shape[1], dtype=np.longdouble)
    for r in range(block.shape[0]):
        s += block[r]
    return s.astype(np.float64)


def _vst(x: np.ndarray) -> np.ndarray:
    """``log(sqrt(x) + sqrt(x + 1))`` with the C library ``log`` (R's); integer counts are looked
    up in a table of their values."""
    import math
    xmax = float(x.max()) if x.size else 0.0
    if xmax < 1e6 and np.array_equal(x, np.floor(x)) and x.min() >= 0:
        k = np.arange(int(xmax) + 1, dtype=np.float64)
        tab = np.array([math.log(v) for v in (np.sqrt(k) + np.sqrt(k + 1.0)).tolist()])
        return tab[x.astype(np.int64)]
    return np.log(np.sqrt(x) + np.sqrt(x + 1.0))


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """``ks.test(x, y)$statistic`` (two-sided): R's cumulative sum of +1/n.x and -1/n.y in the
    order of ``c(x, y)``, kept at the last of tied values."""
    nx, ny = x.size, y.size
    w = np.concatenate([x, y])
    o = np.argsort(w, kind="stable")
    steps = np.where(o < nx, 1.0 / nx, -1.0 / ny)
    z = np.cumsum(steps.astype(np.longdouble)).astype(np.float64)
    ws = w[o]
    if np.unique(w).size < nx + ny:
        keep = np.concatenate([np.flatnonzero(np.diff(ws) != 0), [nx + ny - 1]])
        z = z[keep]
    return float(np.max(np.abs(z)))


# ------------------------------------------------------------------------------------------------
# genes to bins


def convert_bins_hg20(rna: np.ndarray, symbols: np.ndarray, full_anno: pd.DataFrame,
                      dna: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """R ``convert.all.bins.hg20``: per 220 kb bin, the median over the genes whose centre lies in
    ``[start, end]`` (start = the previous bin's end, across chromosome boundaries as in R); an
    empty bin takes the values of the nearest non-empty bin (the earlier one on a tie)."""
    DNA = dna.loc[dna["chrom"] != 24].reset_index(drop=True)
    end = DNA["chrompos"].to_numpy(np.float64)
    start = np.concatenate([[0.0], end[:-1]])
    bchr = DNA["chrom"].to_numpy(np.int64)
    sym_row = pd.Series(np.arange(symbols.size), index=pd.Index(symbols))
    fa = full_anno.loc[full_anno["hgnc_symbol"].isin(sym_row.index)]
    cent = 0.5 * (fa["start_position"].to_numpy(np.float64) + fa["end_position"].to_numpy(np.float64))
    gchr = fa["chromosome_name"].to_numpy(np.int64)
    grow = sym_row.loc[fa["hgnc_symbol"].to_numpy()].to_numpy()
    pb, pr = [], []
    for ch in np.unique(bchr):
        bi = np.flatnonzero(bchr == ch)
        gi = np.flatnonzero(gchr == ch)
        if gi.size == 0:
            continue
        c = cent[gi]
        s_, e_ = start[bi], end[bi]
        # candidate bins: those with end >= c, scanning a few onwards for boundary/overlap cases
        j0 = np.searchsorted(e_, c, side="left")
        for off in range(0, 3):
            j = j0 + off
            ok = j < bi.size
            jj = np.where(ok, j, 0)
            hit = ok & (c <= e_[jj]) & (c >= s_[jj])
            pb.append(bi[jj[hit]]); pr.append(grow[gi[hit]])
        # bins whose start lies above the previous end (first bin of a chromosome) can only
        # be hit at j0; bins with start < previous end are not produced by the table
    pb = np.concatenate(pb) if pb else np.empty(0, dtype=np.int64)
    pr = np.concatenate(pr) if pr else np.empty(0, dtype=np.int64)
    pairs = np.unique(np.column_stack([pb, pr]), axis=0) if pb.size else np.empty((0, 2), np.int64)
    nb = DNA.shape[0]
    n_cells = rna.shape[1]
    out = np.full((nb, n_cells), np.nan)
    cnt = np.bincount(pairs[:, 0], minlength=nb) if pairs.size else np.zeros(nb, np.int64)
    order = np.argsort(pairs[:, 0], kind="stable")
    pairs = pairs[order]
    first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    for m in np.unique(cnt[cnt > 0]):
        b = np.flatnonzero(cnt == m)
        rows = pairs[first[b][:, None] + np.arange(m)[None, :], 1]          # (len(b), m)
        out[b] = np.median(rna[rows], axis=1)
    empty = np.flatnonzero(cnt == 0)
    if empty.size:
        Rw = np.flatnonzero(cnt > 0)
        k = np.searchsorted(Rw, empty)
        left = Rw[np.clip(k - 1, 0, Rw.size - 1)]
        right = Rw[np.clip(k, 0, Rw.size - 1)]
        use = np.where(np.abs(left - empty) <= np.abs(right - empty), left, right)
        out[empty] = out[use]
    return out, DNA[["chrom", "chrompos", "abspos"]]


# ------------------------------------------------------------------------------------------------
# the pipeline


def run(counts, genes, cells, *, id_type="S", cell_line=False, ngene_chr=5, min_gene_per_cell=200,
        LOW_DR=0.05, UP_DR=0.1, win_size=25, norm_cell_names=None, KS_cut=0.1, distance="euclidean",
        genome="hg20", seed=1234, mc=1000, n_jobs=1, keep_intermediate=False) -> CopykatRun:
    """R ``copykat()`` (n.cores = 1). ``counts``: genes x cells (dense or sparse)."""
    inter: dict = {}
    rng = RRandom(seed)
    X = sparse.csc_matrix(counts, dtype=np.float64)
    genes = np.asarray(genes).astype(str)
    cells = np.asarray(cells).astype(str)
    log.info("step 1: read and filter data; %d genes, %d cells", X.shape[0], X.shape[1])
    gene_raw = np.diff(X.indptr) if X.has_canonical_format else (X != 0).sum(axis=0).A1
    X.eliminate_zeros()
    gene_raw = np.diff(X.indptr)
    if np.sum(gene_raw > min_gene_per_cell) == 0:
        raise ValueError("none cells have more than min.gene.per.cell")
    if np.sum(gene_raw < min_gene_per_cell) > 1:
        keep = gene_raw >= min_gene_per_cell
        X, cells = X[:, keep], cells[keep]
    raw_cells = cells.copy()
    der = np.bincount(X.indices, minlength=X.shape[0]) / X.shape[1]
    if np.sum(der > LOW_DR) >= 1:
        keep = der > LOW_DR
        X, genes = X[keep], genes[keep]
    WNS1 = "data quality is ok"
    if X.shape[0] < 7000:
        WNS1 = "low data quality"
        UP_DR = LOW_DR
    log.info("step 2: annotations gene coordinates")
    anno, rows = annotate(genes, id_type, genome)
    X = X[rows]
    if genome == "hg20":
        cyc = set(load_hg20_cycle_genes())
        sym = anno["hgnc_symbol"].astype(str)
        drop = sym.str.match(r"^HLA-").to_numpy() | sym.isin(cyc).to_numpy()
        anno, X = anno.loc[~drop].reset_index(drop=True), X[~drop]
    chrom = anno["chromosome_name"].to_numpy()
    chrom_codes = pd.factorize(chrom)[0]
    n_chrom = int(np.unique(chrom).size)
    rm2 = _chrom_runs_filter(None, X, chrom_codes, n_chrom, ngene_chr)
    if rm2.all():
        raise ValueError("all cells are filtered")
    X, cells = X[:, ~rm2], cells[~rm2]
    raw3 = X.toarray()
    norm = _vst(raw3)
    norm = norm - r_means(norm, axis=0)[None, :]
    log.info("step 3: smoothing data with dlm")
    sm = kalman_smooth_matrix(norm, dV=0.16, dW=0.001)
    smooth = sm - r_means(sm, axis=0)[None, :]
    if keep_intermediate:
        inter.update(smooth=smooth, smooth_cells=list(cells), anno=anno.copy())
    log.info("step 4: measuring baselines")
    cl_cells = list(cells)
    if cell_line:
        relat, rel_cells, CL = baseline_synthetic(smooth, list(cells), rng)
        WNS, preN = "run with cell line mode", []
        CL_names = list(cells)
        # columns are now grouped by cluster
        relat_cells = rel_cells
    elif norm_cell_names is not None and len(norm_cell_names) > 1:
        known = np.isin(cells, list(norm_cell_names))
        if known.sum() == 0:
            raise ValueError("known normal cells provided; however none existing in testing dataset")
        basel = r_medians(smooth[:, known], axis=1)
        fit = hclust(euclidean_dist(smooth), "ward.D2")
        CL = _cut_min_size(fit, 6, 10)
        CL_names = list(cells)
        WNS, preN = "run with known normal", list(norm_cell_names)
        relat = smooth - basel[:, None]
        relat_cells = list(cells)
    else:
        basel, WNS, preN, CL, info = baseline_norm_cl(smooth, list(cells), 10)
        CL_names = list(cells)
        inter.update(baseline=info, bnc_preN=list(preN), bnc_WNS=WNS, bnc_cl=CL.copy())
        if WNS == "unclassified.prediction":
            g = baseline_gmm(smooth, list(cells))
            if g is not None:
                basel, preN = g
                WNS = ""
                inter["gmm_preN"] = list(preN)
        relat = smooth - basel[:, None]
        relat_cells = list(cells)
    DR2 = np.bincount(X.indices, minlength=X.shape[0]) / X.shape[1]
    gkeep = DR2 >= UP_DR
    relat = relat[gkeep]
    anno2 = anno.loc[gkeep].reset_index(drop=True)
    X2 = X[gkeep]
    rm3 = _chrom_runs_filter(None, X2, chrom_codes[gkeep], n_chrom, ngene_chr)
    rm3_names = set(cells[rm3])
    if len(rm3_names) == relat.shape[1]:
        raise ValueError("all cells are filtered")
    keep_cols = np.array([c not in rm3_names for c in relat_cells])
    relat = relat[:, keep_cols]
    relat_cells = [c for c, k in zip(relat_cells, keep_cols) if k]
    CLs = pd.Series(CL, index=CL_names)
    CLs = CLs[CLs.index.isin(relat_cells)]
    CLs = CLs.loc[relat_cells]
    clu = CLs.to_numpy()
    log.info("step 5: segmentation")
    gamma = GammaStream(rng)
    logCNA, BR = cna_mcmc(clu, relat, win_size, KS_cut, gamma, mc)
    if BR.size < 25:
        log.info("too few breakpoints detected; decreased KS.cut to 50%")
        logCNA, BR = cna_mcmc(clu, relat, win_size, 0.5 * KS_cut, gamma, mc)
    if BR.size < 25:
        log.info("too few breakpoints detected; decreased KS.cut to 75%")
        logCNA, BR = cna_mcmc(clu, relat, win_size, 0.5 * 0.5 * KS_cut, gamma, mc)
    if BR.size < 25:
        raise ValueError("too few segments; try to decrease KS.cut; or improve data")
    results_com = logCNA - r_means(logCNA, axis=0)[None, :]
    if keep_intermediate:
        inter.update(breaks=BR, logCNA=logCNA, relat_cells=list(relat_cells), clu=clu)
    sym2 = anno2["hgnc_symbol" if genome == "hg20" else "mgi_symbol"].to_numpy().astype(str)
    if genome == "hg20":
        log.info("step 6: convert to genomic bins")
        uber, pos_cols = convert_bins_hg20(results_com, sym2, load_genome_annotation("hg20"), load_hg20_bins())
        if keep_intermediate:
            inter["bins"] = uber
    else:
        uber = results_com
        pos_cols = anno2.iloc[:, :7]
    if cell_line:
        tree = hclust(_cluster_dist(uber, distance), "ward.D")
        cna = pd.concat([pos_cols.reset_index(drop=True), pd.DataFrame(uber, columns=relat_cells)], axis=1)
        return CopykatRun(prediction=None, cna=cna, hclustering=tree, cells=relat_cells, WNS1=WNS1,
                          WNS=WNS, preN=[], intermediate=inter)
    log.info("step 7: re-adjust baseline")
    pred, mat_adj, tree, pinfo = _predict(uber, results_com, relat_cells, preN, distance, WNS,
                                          strict=(genome != "hg20"))
    inter["classification"] = pinfo
    ndef = [c for c in raw_cells if c not in set(relat_cells)]
    res = pd.DataFrame({"cell.names": list(relat_cells) + ndef,
                        "copykat.pred": list(pred) + ["not.defined"] * len(ndef)})
    cna = pd.concat([pos_cols.reset_index(drop=True), pd.DataFrame(mat_adj, columns=relat_cells)], axis=1)
    return CopykatRun(prediction=res, cna=cna, hclustering=tree, cells=relat_cells, WNS1=WNS1, WNS=WNS,
                      preN=list(preN), intermediate=inter)


def _predict(uber, results_com, cells, preN, distance, WNS, strict):
    """Steps 7-8 of R copykat (v1.2.x): cut the cell tree, take the clusters holding most of the
    pre-defined diploid cells as diploid, re-adjust the baseline, cluster again and assign each
    cluster by its Wasserstein distance to the diploid and aneuploid consensus profiles."""
    preN_set = set(preN)
    nP = len(preN)

    def cl_id(cut):
        return np.array([sum(1 for c in np.asarray(cells)[cut == i] if c in preN_set) / nP
                         for i in range(1, int(cut.max()) + 1)])

    tree = hclust(_cluster_dist(uber, distance), "ward.D")
    k_short = _final_k(tree)
    hc_short = tree.cutree(k_short)
    ids = cl_id(hc_short)
    dip = np.isin(hc_short, np.flatnonzero(ids == ids.max()) + 1)
    rat = uber - r_means(uber[:, dip], axis=1)[:, None]
    rat = rat - r_means(rat, axis=0)[None, :]
    norm = rat[:, dip]
    cf_h = r_sds(norm, axis=1)
    base = r_means(norm, axis=1)
    colmean = r_means(rat, axis=0)
    adj = np.where(np.abs(rat - base[:, None]) <= (0.25 * cf_h)[:, None], colmean[None, :], rat)
    mat_adj = adj - r_means(adj, axis=0)[None, :]
    tree = hclust(_cluster_dist(mat_adj, distance), "ward.D")
    k_final = _final_k(tree)
    hc = tree.cutree(k_final)
    ids = cl_id(hc)
    cl_dip = np.flatnonzero(ids == ids.max()) + 1
    cl_ane = np.flatnonzero(ids == ids.min()) + 1
    cons_dip = r_medians(results_com[:, np.isin(hc, cl_dip)], axis=1)
    cons_ane = r_medians(results_com[:, np.isin(hc, cl_ane)], axis=1)
    r = float(np.corrcoef(cons_dip, cons_ane)[0, 1])
    pred = np.empty(len(cells), dtype=object)
    if (r > 0.6) if strict else (r >= 0.6):
        pred[:] = "diploid"
    else:
        sd_dip, sd_ane = np.sort(cons_dip), np.sort(cons_ane)
        for i in range(1, int(hc.max()) + 1):
            cons = np.sort(r_medians(results_com[:, hc == i], axis=1))
            w_dip = r_means(np.abs(sd_dip - cons)[:, None], 0)[0]
            w_ane = r_means(np.abs(sd_ane - cons)[:, None], 0)[0]
            pred[hc == i] = "diploid" if w_dip < w_ane else "aneuploid"
    info = dict(k_first=k_short, k_final=k_final, cl_id=ids.tolist(), cor=r, clusters=hc)
    if WNS == "unclassified.prediction":
        pred = np.where(pred == "diploid", "c1:diploid:low.conf",
                        np.where(pred == "aneuploid", "c2:aneuploid:low.conf", pred))
    return pred, mat_adj, tree, info


def annotate(genes: np.ndarray, id_type: str, genome: str) -> tuple[pd.DataFrame, np.ndarray]:
    """R ``annotateGenes.hg20``/``mm10`` then ``order(abspos)``: the annotation rows of the genes
    found (first entry per symbol), in the data's gene order, then stably by ``abspos``. Returns
    the annotation and the matching row indices of ``genes``."""
    full = load_genome_annotation(genome)
    sym_col = "hgnc_symbol" if genome == "hg20" else "mgi_symbol"
    key = "ensembl_gene_id" if str(id_type)[:1] in ("E", "e") else sym_col
    first = ~pd.Index(genes).duplicated()
    in_anno = pd.Index(genes).isin(full[key].astype(str)) & first
    anno = full.loc[full[key].astype(str).isin(set(genes[in_anno]))]
    anno = anno.loc[~anno[sym_col].duplicated()]
    pos = pd.Series(np.arange(genes.size), index=genes)[~pd.Index(genes).duplicated()]
    rows = pos.loc[anno[key].astype(str).to_numpy()].to_numpy()
    o = np.argsort(rows, kind="stable")                      # data order
    anno, rows = anno.iloc[o].reset_index(drop=True), rows[o]
    o = np.argsort(anno["abspos"].to_numpy(np.float64), kind="stable")
    return anno.iloc[o].reset_index(drop=True), rows[o]
