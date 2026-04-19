# pycopykat vs R copykat — 17-patient benchmark findings

## TL;DR (academically precise framing)

**Clustering quality is essentially identical.** Label-agnostic mean
accuracy — `max(acc, 1-acc)` per patient — is **0.933 (R) vs 0.923 (py)**,
a 1-percentage-point difference within noise. When ARI is high and the
two predictions partition the cells the same way, label-agnostic accuracy
converges to the standard accuracy.

**Signed accuracy diverges in four patients because R assigns the
`diploid` / `aneuploid` labels to the wrong clusters**, even though it
finds the same partition. The root cause is baseline-cluster selection:
R copykat picks the Ward cluster with the smallest F-test sigma as the
diploid reference, and in these datasets the tumor cluster happens to
have the tighter sigma than the stromal cluster. pycopykat's
tied-covariance 3-GMM gives a different — and in these cases correct —
minimum-sigma cluster.

| metric | R copykat | pycopykat |
|---|---|---|
| mean accuracy vs 3CA label (signed) | 0.749 | **0.900** |
| mean **label-agnostic** accuracy (clustering only) | **0.933** | 0.923 |
| mean py↔R ARI | — | 0.878 |
| mean py↔R FMI | — | 0.949 |
| mean runtime (8 cores) | 11.9 min | 2.9 min |
| mean speedup | — | **~4×** |

## 17-patient outcome classification

| mechanism | count | description |
|---|---|---|
| **matched** | 12/17 | Both methods produce near-identical predictions (κ ≥ 0.94) |
| **R_label_flipped (py correct)** | 4/17 | Same partition (ARI 0.91–0.97) but R assigns `diploid`/`aneuploid` labels inversely. pycopykat's baseline selection is correct. Patients: Lee/SMC16, Obradovic/Patient5, Obradovic/Patient2, Qian/12 |
| **different_clusters** | 1/17 | Structural disagreement (ARI 0.18). Kim/P1028 has 95% tumor purity and is an intrinsic copykat edge case (no reliable normal reference to anchor the baseline). |

## What the results do and do not support

**Supported:**

* pycopykat implements the copykat algorithm faithfully — 12 of 17
  patients show near-identical outputs to R, including all four Gao
  breast patients with κ ≥ 0.99.
* pycopykat runs ~4× faster on 8 cores on every patient tested.
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
not an implementation defect in either R or py.

## Per-patient table

See `label_agnostic_summary.csv` in the same directory for the full
per-patient breakdown with signed + label-agnostic accuracy, ARI, κ,
FMI, and mechanism classification.
