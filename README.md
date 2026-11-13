# pycopykat

A Python implementation of [CopyKAT](https://github.com/navinlabcode/copykat) (Gao et al., *Nat Biotechnol* 2021): copy-number profiles of single cells in 220 kb bins and the call of each cell as aneuploid (tumour) or diploid (normal) from scRNA-seq UMI counts.

**Status:** v0.2.0 — reproduces R copykat 1.2.5. With the same seed pycopykat gives R's breakpoints, R's baseline cells and R's call for every cell; the copy-number matrices agree to about 1e-10. On six breast tumours of the CopyKAT paper it ran 3.5 to 14 times faster than R on one thread. See *Reproducing R* below and [`report/`](report/).

## Installation

```bash
pip install pycopykat
```

Python 3.10–3.12. The wheel bundles R copykat's reference tables (hg20 and mm10 gene positions, the 220 kb bins, the cell-cycle genes), identical to the package's `sysdata.rda`.

## Usage

```python
import pandas as pd
from pycopykat import CopykatConfig, copykat

counts = pd.read_csv("counts.tsv", sep="\t", index_col=0)     # genes x cells, raw UMI counts
res = copykat(counts, CopykatConfig(id_type="Symbol", genome="hg20", n_jobs=8))

res.prediction        # cell.names, copykat.pred  (as R's res$prediction)
res.cna_mat           # bins x cells, index (chrom, chrompos, abspos)  (R's res$CNAmat)
res.hclustering       # R hclust of the cells (merge, height, order)
res.to_txt("out/", "sample")
```

`copykat` also takes an `AnnData` (cells x genes, counts in `X`). The arguments of `CopykatConfig` are those of R `copykat()` with the same defaults: `id_type` (`"Symbol"`/`"Ensembl"`), `genome` (`"hg20"`/`"mm10"`), `cell_line`, `ngene_chr`, `min_gene_per_cell`, `low_dr`, `up_dr`, `win_size`, `ks_cut`, `distance` (`"euclidean"`, `"pearson"`, `"spearman"`) and `norm_cell_names` (known normal cells); `seed` is the value R copykat passes to `set.seed` (1234), and `n_jobs` sets the threads of the steps without random numbers (results do not depend on it).

Per patient on an AnnData:

```python
from pycopykat import copykat_by_batch
adata = copykat_by_batch(adata, batch_key="patient", config=CopykatConfig(n_jobs=8))
# adata.obs["copykat_pred"], adata.obsm["copykat_cna"], adata.uns["copykat_bins"]
```

Command line: `pycopykat run --input counts.tsv --output-dir out/ --sam-name sample`, or `pycopykat run-h5ad --adata data.h5ad --batch-key patient --output out.h5ad`.

In OmicVerse: `ov.single.CNV(adata, method="copykat").run()`.

## Reproducing R

pycopykat follows R copykat 1.2.5 (`copykat()` and its helpers) step by step, in R's order of genes and cells:

* **Filters and annotation** — the cell and gene filters, the annotation (first entry per symbol, data order, then stably by `abspos`), the removal of HLA and cell-cycle genes and the per-cell chromosome-coverage test (runs of chromosomes over the expressed genes) are R's.
* **Smoothing** — `log(sqrt(x) + sqrt(x + 1))`, centring, and the local-level Kalman smoother of `dlm` (`dlmModPoly(1, dV = 0.16, dW = 0.001)`); dlm computes on SVD factors, which shifts the first genes of each cell by about 1e-10.
* **Baseline** — `hclust(ward.D2)` and `cutree` are ported from R's Fortran and C (nearest-neighbour list, tie order, cluster numbering); mixtools' `normalmixEM` (three components, common sigma) is ported with its `normpost` routine, so the EM takes the same iterations; the silhouette, the F test and the GMM fallback (`baseline.GMM`) follow R. Known normal cells and the cell-line mode (`baseline.synthetic`, `set.seed(123)` and `rnorm` per cluster) are supported.
* **Segmentation** — `CNA.MCMC` draws `MCpoissongamma`, i.e. `rgamma(1000, shape, rate)`, for every window of the cluster medians and for every segment of every cell. R's Mersenne-Twister, `norm_rand` (inversion) and `rgamma` (Ahrens–Dieter GD and GS, `exp_rand`) are ported (`pycopykat._rcompat`); the draws are produced in vectorised form and are identical to R's, so the KS breakpoints and the posterior means are R's. `ks.test` uses R's cumulative sum, and the KS cut-off is lowered as in R when fewer than 25 breakpoints are found.
* **Bins and calls** — `convert.all.bins.hg20` (gene centres in `[start, end]` with the start taken from the previous bin, empty bins filled from the nearest bin), the v1.2 classification (`ward.D` trees cut at k = 4, 3 or 2, the clusters holding the pre-defined diploid cells, the baseline re-adjustment, the correlation rule and the Wasserstein assignment) and the low-confidence labels are R's.

R copykat with `n.cores > 1` does not repeat its own results (`mclapply` children start from unseeded generators); pycopykat always follows the `n.cores = 1` stream.

Comparison with R copykat 1.2.5 (same seed, `n.cores = 1`):

| sample (GSE148673) | cells | breakpoints | calls identical | max \|ΔCNA\| | pycopykat (s) | R (s) |
|---|---|---|---|---|---|---|
| DCIS1 | 1,480 | 231 = 231 | yes | 4.3e-11 | see report | 728 |
| TNBC1 | 1,097 | 291 = 291 | yes | 5.2e-11 | | 488 |
| TNBC2 | 1,034 | 267 = 267 | yes | 5.5e-11 | | 439 |
| TNBC3 | 532 | 326 = 326 | yes | 5.3e-11 | | 222 |
| TNBC4 | 3,056 | 252 = 252 | yes | 6.2e-11 | | 2,449 |
| TNBC5 | 3,225 | 219 = 219 | yes | 1.3e-11 | | 2,703 |

The known-normal, cell-line and Pearson-distance modes were checked the same way on TNBC3. `tests/test_parity.py` checks copykat's example data (`exp.rawdata`) against stored R results, and `tests/test_rcompat.py` the R pieces against stored R values; `tests/r_reference.R` and `tests/r_components.R` regenerate them.

## Testing

```bash
pip install -e '.[dev]'
pytest
```

## Citation

Gao R, Bai S, Henderson YC, et al. Delineating copy number and clonal substructure in human tumors from single-cell transcriptomes. *Nat Biotechnol* 39, 599–608 (2021).

## License

GPL-2.0-or-later, as R copykat (GPL-2).
