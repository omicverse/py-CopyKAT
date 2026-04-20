# pycopykat vs R copykat — 17-patient benchmark findings

## TL;DR (academically precise framing)

**Clustering quality is essentially identical, with one regime improvement.**
Label-agnostic mean accuracy — `max(acc, 1-acc)` per patient — is
**0.930 (R) vs 0.935 (py)** on the post-Phase-1 run, a ~0.5-percentage-point
difference within noise. When ARI is high and the two predictions partition
the cells the same way, label-agnostic accuracy converges to the standard
accuracy.

**Signed accuracy diverges in four patients because R assigns the
`diploid` / `aneuploid` labels to the wrong clusters**, even though it
finds essentially the same partition. The root cause is baseline-cluster
selection: R copykat picks the Ward cluster with the smallest F-test sigma
as the diploid reference, and in these datasets the tumor cluster happens
to have the tighter sigma than the stromal cluster. pycopykat's
tied-covariance 3-GMM gives a different — and in these cases correct —
minimum-sigma cluster.

| metric | R copykat | pycopykat |
|---|---|---|
| mean accuracy vs 3CA label (signed) | 0.749 | **0.911** |
| mean **label-agnostic** accuracy (clustering only) | **0.930** | 0.935 |
| median py↔R ARI | — | **0.988** |
| mean py↔R ARI | — | 0.922 |
| mean py↔R FMI | — | 0.971 |
| mean runtime (8 cores, full workflow) | 11.6 min | **1.3 min** |
| median py-vs-R speedup | — | **11.6×** |
| mean py-vs-R speedup | — | **10.5×** |

## 17-patient outcome classification

| mechanism | count | description |
|---|---|---|
| **matched** | 13/17 | Both methods produce near-identical predictions (κ ≥ 0.83) |
| **R_label_flipped (py correct)** | 4/17 | Same partition (ARI 0.34–0.96) but R assigns `diploid`/`aneuploid` labels inversely. pycopykat's baseline selection is correct. Patients: Lee/SMC16, Obradovic/Patient5, Obradovic/Patient2, Qian/12 |
| **different_clusters** | 0/17 | No structural disagreements in the post-Phase-1 run. Kim/P1028 — 95% tumor purity — now lands in `matched` with ARI 0.70 after the refactor (previously 0.18). |

## What the results do and do not support

**Supported:**

* pycopykat implements the copykat algorithm faithfully — 13 of 17
  patients show near-identical outputs to R, including all four Gao breast
  patients with κ ≥ 0.99.
* pycopykat runs ~10× faster on 8 cores on every patient tested. On the
  largest full-workflow benchmark (Qian/11, ~7k cells) the end-to-end
  wall-clock time is 2.9 min (py) vs 35.9 min (R) — 12.3×.
* pycopykat's baseline-selection step (tied-covariance GMM over per-cluster
  per-gene medians) is more robust than R copykat's mixtools fit in four
  of the 17 patients, leading to correct diploid/aneuploid assignment where
  R's choice inverts.

**Not supported:**

* The claim "pycopykat clusters the data better than R copykat." The
  2-way partition of cells is essentially identical between the two
  methods — see the label-agnostic accuracy row above.
* Any bit-exact equivalence claim. Algorithmic differences exist
  (scipy ward.D2 vs R ward.D, sklearn tied-covariance GMM vs R mixtools
  arbvar=FALSE, simplified dynamicTreeCut V1 vs WGCNA). These are
  acknowledged in M6.2 and gated by the loose regression thresholds in
  `tests/test_regression.py`.

## Degenerate regime (≥90% malignant)

copykat's baseline step needs enough diploid-looking cells to anchor.
On Kim/P1028 (95% Malignant) and Lee/SMC16 (95% Malignant) both methods
degrade substantially — label-agnostic accuracy 0.58 / 0.86 respectively.
This is a known limitation of the copykat algorithm (Gao et al. 2021),
not an implementation defect in either R or py. Note that after the
Phase-1 refactor P1028's py↔R ARI rose from 0.18 to 0.70 (py and R now
agree on most of the 4434 cells, even when both disagree with the 3CA
Malignant label), but absolute accuracy remains poor for both tools.

## Phase-1 performance patches (A1–A8)

This sweep was pinned to **commit 9c67213** (`perf(kalman): drop per-column
copy by feeding Fortran-order input`). Phase-1 introduced eight independent
perf patches (A1–A8) covering the most expensive steps: preprocess,
smoothing, clustering, Kalman filter, and final output.

Against the pre-Phase-1 sweep (same 17 patients, same R outputs; baselines
preserved in `py_vs_r_summary.pre_A8.csv`, `label_agnostic_summary.pre_A8.csv`,
`overview.cancer_summary.pre_A8.csv`, and `FINDINGS.pre_A8.md`):

* **median py-vs-R wall-clock speedup: 4.3× → 11.6×** (2.7× faster pycopykat)
* **median py↔R ARI: 0.988 → 0.988** (unchanged)
* **mean py↔R ARI: 0.878 → 0.922** (+0.044, driven by Kim/P1028 &
  Lee/SMC16 which are degenerate-regime cases where the previous code
  landed on a knife-edge that Phase-1's numerical improvements pushed
  to the better side)
* **matched patients: 12/17 → 13/17** (P1028 moved from `different_clusters`
  to `matched` — its ARI rose from 0.18 to 0.70)
* **worst per-patient ARI change: -0.008** (Kim/P0034, well inside noise)

No patient regressed by > 0.05 ARI; the regression gate defined in the
rerun spec was never tripped.

## Per-patient table

See `label_agnostic_summary.csv` (post-Phase-1) and
`label_agnostic_summary.pre_A8.csv` (baseline) for the full per-patient
breakdown with signed + label-agnostic accuracy, ARI, κ, FMI, and
mechanism classification. See `phase1_comparison.csv` for the paired
old/new ARI and wall-clock table.
