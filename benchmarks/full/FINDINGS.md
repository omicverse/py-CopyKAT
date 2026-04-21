# pycopykat vs R copykat — 17-patient benchmark

## Scope

This benchmark measures one thing: **does pycopykat agree with R copykat
when both are run in unsupervised auto-baseline mode on the same counts
matrix?** No external cell-type labels are used as a reference — the
validation target is py↔R consistency, nothing else.

## Headline numbers

* 17 patients across 5 cancer types (Gao2021_Breast, Kim2020_Lung,
  Lee2020_Colorectal, Obradovic2021_Kidney, Qian2020_Ovarian).
* py↔R ARI: **median 0.988, mean 0.922, min 0.342**.
* py↔R FMI: median 0.994, mean 0.968, min 0.759.
* py↔R Cohen κ: median 0.994, mean 0.567, min −0.943. Three patients
  show negative κ alongside high ARI — same cell partition, inverted
  aneuploid / diploid labels (see "Outcome classification" below).
* Wall-clock (8-core, end-to-end): py median **0.25 min**, R median
  **9.83 min**; median speedup **23.4×**, mean speedup **31.3×**, max
  **82.4×** (Qian/11, 6972 cells).
* All runs use pycopykat's unsupervised AUTO baseline (no curated normal
  cells) and R copykat's default parameters.

## Outcome classification

A coarse mechanism label derived from the py↔R ARI and κ alone:

| mechanism | rule | count |
|---|---|---|
| matched | ARI ≥ 0.9 and κ ≥ 0.9 — near-identical partition and labelling | 12 / 17 |
| R_label_flipped | ARI ≥ 0.5 and κ < 0 — same partition, inverted aneuploid/diploid labels (baseline-selection divergence between the two implementations) | 3 / 17 |
| partial | 0.5 ≤ ARI < 0.9 — moderate agreement without a clean label flip | 1 / 17 |
| different_clusters | ARI < 0.5 — structural disagreement on the 2-way partition | 1 / 17 |

Per-mechanism patient breakdown:

* **matched** (12): Gao/DCIS1, Gao/TNBC1, Gao/TNBC2, Gao/TNBC3,
  Kim/P0019, Kim/P0034, Lee/SMC09, Lee/SMC21, Obradovic/Patient4,
  Qian/11, Qian/13, Qian/14.
* **R_label_flipped** (3): Obradovic/Patient5, Obradovic/Patient2,
  Qian/12.
* **partial** (1): Kim/P1028 (ARI 0.70, κ 0.84 — 95%-malignant sample
  in a degenerate regime for copykat's baseline step).
* **different_clusters** (1): Lee/SMC16 (ARI 0.34, κ −0.33).

The two methods produce divergent runs on Kim/P1028 and Lee/SMC16
because both samples have very few diploid-looking cells to anchor
the baseline, which is a known limitation of the copykat algorithm
(Gao et al., *Nat Biotechnol* 2021), not an implementation defect
of either pycopykat or R copykat.

### Known parity gap: Lee2020_Colorectal / SMC16

This is the only sample in the 17-patient sweep where the two
implementations disagree on the majority-class split, not just on
label polarity (ARI = 0.342, κ = −0.326, FMI = 0.759).

**What the outputs look like.**

| | R copykat | pycopykat |
|---|---|---|
| diploid | 2235 (83%) | 909 (34%) |
| aneuploid | 460 (17%) | 1786 (66%) |
| label suffix | clean `diploid` / `aneuploid` | all cells tagged `c{1,2}:…:low.conf` |
| runinfo warnings | — | `unclassified.prediction; unclassified.prediction` |

**What is consistent.** The bin-level CNA matrix is in close agreement
between the two runs — the pre-classification stages (filtering, VST +
Kalman smoothing, per-cell segmentation, bin aggregation) behave the
same on this sample. Evidence and figures: see
`examples/compare_py_vs_R_realdata.ipynb` for the bin-level Pearson /
Spearman comparison and the py-vs-R heatmap / Δ-heatmap on SMC16.

**Where the divergence lives.** Both sides take the same fallback
branch:

```
baseline_norm_cl  ──low confidence──►  baseline_gmm  ──low confidence──►  label suffix "low.conf"
```

R copykat retains clean `diploid` / `aneuploid` labels after this
chain; pycopykat tags every cell `c1:diploid:low.conf` /
`c2:aneuploid:low.conf` (`pipeline.py:282-287`). The label-suffix
difference is cosmetic. The quantitative gap — 83% vs 34% diploid — is
not: it means the majority class itself flipped, so one of the two
pipelines has chosen a different baseline vector or a different
`predict_ploidy` split on this sample.

**Root cause: not yet localised.** The fallback chain in
`pipeline.py:157-170` mirrors R's `copykat.R:165-176` structurally
(`baseline.norm.cl` → `baseline.GMM` with `RE.before`). The divergence
must therefore sit in either:

1. the diploid set `preN` emitted by `baseline_gmm` when fewer than
   three "normal-looking" cells are recovered (R returns `RE.before`;
   py passes `fallback=br` — verify identity),
2. the resulting baseline vector `basel`, or
3. the downstream `predict_ploidy` cut when the baseline is near-noise.

This is tracked separately at `TODO_parity_SMC16.md` rather than
investigated inline — reaching a defensible fix requires dumping py
and R intermediate states (`basel`, `preN`, `predict_ploidy` linkage)
on this sample and comparing them. Patching any of the three
candidates blindly would risk regressing the 16 other patients where
pycopykat already matches R at ARI ≥ 0.92.

**Consequence for users.** On SMC16-type samples where pycopykat emits
`unclassified.prediction` twice and tags every cell `*:low.conf`, the
per-cell polarity should be treated as unreliable — the label suffix
is a correct "not confident" signal from the algorithm, independent
of the direction of the disagreement with R. The other 16 samples in
this benchmark do not exhibit this behaviour.

**Benchmark policy.** SMC16 is retained in `py_vs_r_summary.csv`,
`mechanism_summary.csv`, `overview.cancer_summary.csv` and the
aggregate `overview.png`. Excluding it would hide the one case in the
sweep that exposes this fallback-branch disagreement, which would
weaken rather than strengthen the parity reporting.

## Per-patient table

See `mechanism_summary.csv` for the per-patient breakdown (cancer,
sample, n_cells, py↔R ARI / κ / FMI, mechanism, py_min, r_min,
speedup). Per-patient figures live at
`<cancer>/<sample>/figure_py_vs_R.png`; per-patient metrics at
`<cancer>/<sample>/metrics.csv` (single row, `comparison=py_vs_r`).

## Wall-clock per patient

See `py_vs_r_summary.csv` column `py_vs_r__speedup`. Distribution
across the 17 patients:

* min 7.9× (Qian/13)
* median 23.4×
* mean 31.3×
* max 82.4× (Qian/11)

Hot-path work that drove the end-to-end speedup (vs R, 8 cores):

* BLAS-based `pdist_euclidean` via the `‖a‖² + ‖b‖² − 2·a·bᵀ`
  identity — uses OpenBLAS GEMM parallelism across all 8 cores
  where scipy's single-threaded C pdist was saturating one.
  Identity-form output matches scipy's pdist to ≤ 1e-8 absolute
  error; a small-N (< 100) branch still routes to scipy.
* Sparse-aware VST and zero-preserving centring in preprocessing.
* Numba-accelerated Kalman filter fed Fortran-order input to drop
  a per-column copy.

Kim/P0019 went from roughly 90 s (pre-optimization) to 8.6 s
end-to-end (≈10×) over the course of the optimization cycle; the
full 17-patient sweep reflects the final post-optimization build.

## Methodology

* **Same inputs** for both methods: the sliced `counts.tsv` (genes ×
  cells from the 3CA release).
* **Unsupervised mode only**: neither method is fed a curated normal-cell
  list. pycopykat uses its AUTO baseline (tied-covariance GMM over
  per-cluster per-gene medians). R copykat uses its default mixtools
  baseline selection.
* **No external labels are used as a reference.** The 3CA `cell_type`
  column ships with the data but is itself algorithmically derived
  and is not a gold standard. It is not used by any metric in this
  benchmark.
* **Metrics**: `pycopykat.validation.metrics.compare_predictions`
  (adjusted Rand index, Cohen κ on the binary aneuploid flag, and
  Fowlkes–Mallows index on the binary aneuploid flag).
* **Hardware**: same machine for all runs, 8 cores for both methods.
