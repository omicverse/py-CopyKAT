# TODO — parity investigation on Lee2020_Colorectal / SMC16

**Status:** open. Not part of any release blocker. Do not patch
`baseline_gmm` or `predict_ploidy` without executing the steps below —
the 16 other patients currently match R at ARI ≥ 0.92 and a blind
change would risk regressing them.

**Symptom.** See `FINDINGS.md` → *Known parity gap: Lee2020_Colorectal
/ SMC16*. CNA matrix agrees between py and R; classification flips the
majority class (R: 83% diploid, py: 34% diploid). Both sides trigger
`unclassified.prediction` twice (norm_cl → GMM fallback).

## Evidence to gather before proposing a fix

Each step should dump a small artefact under
`benchmarks/full/Lee2020_Colorectal/SMC16/parity_probe/` so the fix
can be reviewed against primary data rather than argument.

1. **Identity of `preN` from `baseline_gmm` fallback.** Add a debug
   hook in `pipeline.py:164` to dump `br.preN` (the diploid cell set
   returned by the GMM fallback) on SMC16. Save as
   `py_preN.txt`. Run R copykat with an equivalent hook around
   `copykat.R:172` (`basa$preN` after `baseline.GMM`) and save as
   `r_preN.txt`. Compare set overlap. **If `preN` sets differ
   substantially, the divergence is in `baseline_gmm`.**

2. **Identity of baseline vector `basel`.** Dump `br.basel` (py) and
   `basa$basel` (R) on the same smoothed matrix. Compare element-wise
   correlation and max-abs-diff. **If the vectors are close but
   `preN` differ, the divergence is cosmetic upstream and real
   downstream — proceed to step 3.**

3. **`predict_ploidy` behaviour with R's baseline.** Feed R's `basel`
   + R's `preN` into pycopykat's `predict_ploidy` and check whether
   the py output now matches R's. **If yes, the divergence is
   entirely in `baseline_gmm`'s `preN` / `basel` selection, and the
   fix belongs there. If no, `predict_ploidy` itself is part of the
   gap.**

4. **`baseline_gmm` internal behaviour.** If step 1 localised the gap
   to `baseline_gmm`, trace through
   `pycopykat/baseline/gmm.py` on SMC16:
   - How many cells does each 3-GMM classify as diploid?
   - What does the "< 3 diploids" fallback to `RE.before` look like
     compared to R's `return(RE.before)` branch?
   - Are `mu.cut=0.05`, `Nfraq.cut=0.99`, `max.normal=5` identically
     interpreted?

## What a fix must clear before merging

* All 17 patients re-benchmarked with `scripts/run_all_benchmarks.py`.
* Per-patient ARI on the 12 "matched" patients must stay ≥ 0.98
  (pre-fix range 0.98-1.00).
* Per-patient ARI on the 3 "R_label_flipped" patients must stay ≥
  0.91 (pre-fix range 0.91-0.96; label polarity may flip, κ sign is
  not a constraint).
* SMC16 ARI must rise above 0.5 **and** the improvement must be
  traceable to a step documented in the probe artefacts above, not
  to an unrelated threshold tweak.

## Non-goals

* Removing SMC16 from the benchmark. See `FINDINGS.md` for why it is
  retained.
* Tuning `unclassified.prediction` thresholds (`wn ≤ 0.15`,
  `PDt > 0.05`, `min.cells`) to suppress the warning. The warning is
  a correct signal; suppressing it would mask the underlying data
  quality on this sample.
* Changing the `low.conf` label suffix behaviour. It is intended and
  mirrors R's `copykat.R:419-422`.
