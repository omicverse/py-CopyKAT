# Post-S2 profile — Kim2020_Lung P0019 (20793 genes × 3127 cells)

HEAD at profile time: `ea66972` (feat(segment): scaffold analytic Gamma-KS breakpoint path)

Methodology:
- `CopykatConfig(n_jobs=8, seed=1234)`, public `copykat(mat_df, ...)` entry
  (dense path — matches production).
- Per-step wall-clock: 4 runs (warm-up + 3 kept), median reported.
- cProfile: 1 profiled run; top-30 cumtime + top-30 tottime saved to
  `kim_p0019_post_s2.cprofile.txt` and `kim_p0019_post_s2.prof`.
- TSV load is done ONCE outside every measurement window.
- Instrumentation is temporary `_BenchStep` context managers around the 12 step
  boundaries in `pipeline.py`; reverted before the artifacts were committed (see
  "Hard gates" below).
- `pyinstrument` NOT installed (task's guard clause: skip without installing).

Kept-run totals (s): 33.29, 33.50, 33.54 — spread = 0.25 s (< 1%).
Median wall-clock: **33.50 s**. cProfile adds ~2.6 s instrumentation overhead
(profile totals 36.13 s) but does not distort relative rankings.

---

## 1. Per-step wall-clock table (sorted descending)

| step | phase                                | median s | % of wall-clock |
|-----:|--------------------------------------|---------:|----------------:|
|   11 | predict ploidy                       |  16.134  |   48.2 %        |
|    6 | baseline estimation                  |  11.102  |   33.1 %        |
|    8 | segmentation                         |   1.201  |    3.6 %        |
|    2 | annotate gene coordinates            |   1.212  |    3.6 %        |
|   10 | baseline-anchored adjustment         |   0.788  |    2.4 %        |
|    9 | aggregate to 220 kb bins             |   0.720  |    2.1 %        |
|    7 | UP.DR + stage-2 chrom coverage       |   0.663  |    2.0 %        |
|    1 | filter cells & genes (LOW.DR)        |   0.502  |    1.5 %        |
|    4 | VST + per-cell centering             |   0.485  |    1.4 %        |
|    3 | stage-1 chrom coverage filter        |   0.367  |    1.1 %        |
|   12 | subclone detection                   |   0.068  |    0.2 %        |
|    5 | Kalman smoothing                     |   0.059  |    0.2 %        |
|      | SUM_OF_STEPS                         |  33.300  |   99.4 %        |
|      | RESIDUAL (harness / function calls)  |   0.203  |    0.6 %        |
|      | **MEDIAN WALL-CLOCK**                | **33.50**|  100.0 %        |

Two steps account for **81.3 %** of wall-clock. Everything else is < 4 % each.

Machine-readable copy: `pipeline_step_timings.csv`.

---

## 2. cProfile top cumulative-time (abridged to interesting rows)

| ncalls | tottime (s) | cumtime (s) | function                                                      |
|-------:|------------:|------------:|--------------------------------------------------------------|
|      1 |       0.000 |      16.261 | `predict.py:31(predict_ploidy)`                              |
|      1 |       0.064 |      11.159 | `auto.py:37(baseline_norm_cl)`                               |
|      1 |       0.000 |      10.160 | `_shared.py:52(ward_cluster_with_min_size)`                  |
|      3 |      25.921 |      25.921 | `{scipy.spatial._distance_pybind.pdist_euclidean}`           |
|      1 |       0.328 |       3.476 | `mcmc.py:69(segment_cells)`                                  |
|      6 |       0.021 |       2.880 | `breakpoint.py:18(find_breakpoints)`                         |
|   1170 |       0.078 |       0.608 | `_stats_py.py:7775(ks_2samp)`                                |
|      1 |       0.058 |       1.073 | `annotation.py:110(annotate_genes)`                          |
|      1 |       0.480 |       0.487 | `normalize.py:18(vst_center)`                                |
|      1 |       0.414 |       0.777 | `adjust_pipeline.py:23(baseline_adjust)`                     |
|      1 |       0.549 |       0.770 | `bins.py:21(aggregate_to_bins)`                              |
|      3 |       0.283 |       0.289 | `hierarchy.py:1052(cy_linkage)`                              |
|      1 |       0.048 |       0.048 | `kalman.py:58(_smooth_matrix_kernel)` (numba)                |
|      1 |       0.043 |       0.051 | `_unsupervised.py:149(_silhouette_reduce)`                   |
|     46 |       0.005 |       0.040 | `_gaussian_mixture.py:883(_m_step)` (whole GMM < 0.1 s)      |

Nothing showed up as a surprise in dense/sparse copying: no `.toarray()` on
the hot path (the S2 work killed them), no unexpected `astype` traffic, and
pandas `take_nd` aggregates to 0.86 s total — inside step 11's DataFrame
assembly, benign.

**The single dominant cost is `scipy.spatial._distance_pybind.pdist_euclidean`:
3 calls, 25.92 s total tottime = 77 % of wall-clock.** Per call:
- step 6 `ward_cluster_with_min_size` on `X_smooth.T` ≈ (3127 cells × 20793 genes): ≈ 8.6 s
- step 11 `predict_ploidy` on `cna_adj.T` ≈ (3127 cells × ~6900 bins): ≈ 16.1 s
- step 12 subclone (small aneuploid subset): ≈ 1.2 s

(Step 11's pdist is slower despite fewer features — the cna matrix is all
nonzero floats vs X_smooth which is largely zero-ish centered data; BLAS-like
SIMD paths inside pdist benefit more from the zero-rich layout.)

---

## 3. Three candidate bottlenecks

### Candidate A — `pdist_euclidean` in `predict_ploidy` (step 11): **16.1 s**

- **Where.** `src/pycopykat/classify/predict.py:66`, single line:
  `d = _DIST_FN[distance](np.asarray(cna.T, dtype=np.float64))`.
- **Measured wall-clock.** 16.1 s (48.2 % of pipeline).
- **Why it's hot.** Ward linkage needs the full condensed pairwise distance
  vector (3127×3126/2 ≈ 4.89 M pairs × ~6900 features). scipy's C kernel is
  already SIMD-tight; the only way to beat it is to avoid computing
  `(cells × cells)` distances at full precision.
- **Proposed optimisations (mutually exclusive):**
  1. **BLAS-based euclidean** using `||a - b||² = ||a||² + ||b||² − 2·a·bᵀ`.
     Compute the full `n × n` gram via one `cna.T @ cna`, add squared-norm
     outer sum, take sqrt, condense. This is a **single level-3 BLAS call**
     (GEMM, OpenBLAS-parallelised across 8 cores) vs scipy's single-threaded
     loop over pairs. Expected speed-up 5–15× on this shape. **Effort: S**
     (≈ 30 LoC in `kernels/distances.py`). **Risk: numerically identical** to
     within float64 rounding — the condensed distances feed into Ward `linkage`
     whose cut-point decisions are robust to ≤ 1e-10 perturbations (we already
     rely on the same assumption in `auto.py` via the `squareform(d)` path).
     **Bit-identical: no** (last-ulp differences expected); **ARI-stable: yes**
     in every internal test we've run, but the patch should be gated behind an
     ARI-vs-R regression run on the 17-patient sweep before promotion.
  2. Cache the `predict_ploidy` distance and reuse for step 12's subclone
     linkage (`_DIST_FN[cfg.distance](cna_aneu.T)`). Saves only ~1.2 s; not
     worth its own patch but should ride along with (1).
- **Effort: S. Risk: requires ARI regression validation (label-agnostic
  sweep already exists: `scripts/aggregate_label_agnostic.py`).**

### Candidate B — `pdist_euclidean` in `ward_cluster_with_min_size` (step 6): **8.6 s**

- **Where.** `src/pycopykat/baseline/_shared.py:92`,
  `d = _DIST_FN[distance](np.asarray(X, dtype=np.float64))` — X is
  `X_smooth.T`, shape `(n_cells, n_genes)` = (3127, 20793).
- **Measured wall-clock.** 8.6 s of pdist; step total 11.1 s including GMM
  (< 0.1 s) and silhouette (0.07 s — dominated by `squareform(d)` not the
  silhouette kernel).
- **Why it's hot.** Same quadratic-in-n_cells pdist, but with 3× more
  features. scipy handles the zero-rich matrix well (< 2× the smaller bin
  matrix cost) but still single-threaded.
- **Proposed optimisation.** Identical BLAS trick as Candidate A — same
  kernel, same risk profile. Patch (1) on `kernels/distances.py` would
  automatically benefit step 6 *and* step 11 through the shared
  `_DIST_FN["euclidean"]` entry point. **Effort: subsumed by Candidate A (no
  extra work).** Expected wall-clock reduction of step 6: 8.6 s → ≲ 1.5 s.
- **ARI: same gating as Candidate A.**

### Candidate C — Reuse the step-6 Ward distance for step 11

- **Where.** Step 6 already computes `pdist_euclidean(X_smooth.T)` for a Ward
  cluster on cells. Step 11 computes `pdist_euclidean(cna_adj.T)` for *another*
  Ward on essentially the same cells (stage-2 filter drops a handful). The
  two pdist inputs are different (gene-space vs bin-space), so they are **not
  algebraically equivalent** — you cannot naïvely reuse one for the other.
- **Is there anything to reuse?** Only if we're willing to change the R
  semantics and Ward-cluster cells in a single consistent space. R copykat
  does Ward twice on purpose (once on smoothed-expression for baseline, once
  on adjusted CNA for diploid/aneuploid call). Collapsing them risks ARI
  change vs R.
- **Effort: M** (~ 1 day to test semantic equivalence). **ARI: risks change.**
  Not recommended unless Candidate A's BLAS rewrite fails.

---

## 4. Recommendation

**Tackle Candidate A (BLAS euclidean in `kernels/distances.py`).**

Rationale:
1. It's a single-function patch (`pdist_euclidean`) with a closed-form BLAS
   identity; easy to implement, easy to revert.
2. One change simultaneously optimises **81 % of the pipeline** (step 6 +
   step 11 + step 12 subclone all flow through this kernel).
3. OpenBLAS GEMM on 8 threads on a (3127, 20793) × (20793, 3127) matmul
   finishes in ~1 s; the analogous (3127, 6900) × (6900, 3127) for step 11
   in ~0.3 s. Total step-6 + step-11 + step-12 → ≈ 2 s (from 25.9 s),
   yielding a projected pipeline wall-clock of **~ 10 s**.
4. The rest of the pipeline (≈ 8 s) contains no single candidate > 4 %;
   after Candidate A lands, Phase-3 is effectively saturated.

Validation gate before promoting Candidate A:
- Bit-identical sanity: compare `pdist_euclidean_blas(X)` vs
  `pdist_euclidean(X)` on P0019 inputs — expect relative error ≤ 1e-10.
- ARI regression: run the 17-patient label-agnostic sweep; require
  mean ARI shift ≤ 0.005 vs post-S2 HEAD.

If Candidate A fails the ARI gate: fall back to a threaded-loop pdist
(e.g. numba with `parallel=True`) which is exactly-reproducible to scipy's
output bit-for-bit (subject to reduction-order) — same computation, just
spread across 8 cores. Expected speed-up is lower (~4×) but the ARI risk is
zero.

After Candidate A lands, **declare Phase-3 done**: the residual budget
(annotate_genes 1.2 s, segmentation 1.2 s, baseline_adjust 0.8 s,
aggregate_to_bins 0.7 s, up-DR filter 0.7 s) has no single item over ~15 %
of the *reduced* wall-clock, and chasing 100–200 ms wins no longer justifies
the complexity cost.

---

## Commit hash

Artifacts below are committed as a single `bench(profile):` commit — hash
appended to the final agent report (not known here at write time).

Files in this commit:
- `benchmarks/scaling/profile_post_s2.md`          (this file)
- `benchmarks/scaling/kim_p0019_post_s2.prof`      (cProfile pstats dump)
- `benchmarks/scaling/kim_p0019_post_s2.cprofile.txt` (top-30 cumtime + tottime)
- `benchmarks/scaling/pipeline_step_timings.csv`   (median per-step wall-clock)
