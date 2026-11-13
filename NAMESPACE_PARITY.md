# R copykat 1.2.5 exports and their pycopykat counterparts

| R export | pycopykat | parity |
|---|---|---|
| `copykat` | `pycopykat.copykat` (+ `CopykatConfig`, `CopykatResult`) | same breakpoints, baseline cells and calls; CNA to ~1e-10 |
| `annotateGenes.hg20`, `annotateGenes.mm10` | `pycopykat.core.annotate` | identical rows and order |
| `baseline.norm.cl` | `pycopykat.core.baseline_norm_cl` | identical clusters, sigmas (EM iterations), baseline cells |
| `baseline.GMM` | `pycopykat.core.baseline_gmm` | identical calls (same EM) |
| `baseline.synthetic` | `pycopykat.core.baseline_synthetic` | identical (R's `set.seed(123)` + `rnorm`) |
| `CNA.MCMC` | `pycopykat.core.cna_mcmc` | identical draws (R's `rgamma` stream) |
| `convert.all.bins.hg20` | `pycopykat.core.convert_bins_hg20` | identical bin membership and fill |
| `heatmap.3` | `pycopykat.viz.heatmap.plot_cna_heatmap` | matplotlib heatmap (not a copy) |
| `cal_dist` | — | not ported (`test.emd` option) |

R pieces reproduced in `pycopykat._rcompat`: Mersenne-Twister `set.seed`/`unif_rand`, `norm_rand` (inversion),
`rnorm`, `rgamma` (GD/GS with `exp_rand`), `hclust` (`ward.D`, `ward.D2`) and `cutree`, mixtools `normalmixEM`
(three components, common sigma), `mean`/`sd`/`median` with R's long-double sums, and `ks.test`'s statistic.
