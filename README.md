# pycopykat

A pure-Python re-implementation of [CopyKAT](https://github.com/navinlabcode/copykat) (Gao et al., *Nat Biotechnol* 2021) — aneuploid / diploid classification and bin-level CNA calling from scRNA-seq counts. Drop-in for the scanpy / AnnData ecosystem.

- AnnData-native — drop-in for the scanpy ecosystem (`copykat_by_batch` iterates over a `.h5ad` batch key)
- **No `rpy2`**, no R install — the full CopyKAT pipeline (filter → VST → annotate → Kalman smoothing → KS breakpoints → MCMC segmentation → GMM baseline → Ward clustering → per-cell prediction) is implemented directly in NumPy / SciPy / Numba
- Same function surface as the R workflow (`annotateGenes.hg20` / `annotateGenes.mm10` / `baseline.GMM` / `baseline.norm.cl` / `CNA.MCMC` / `convert.all.bins.hg20` / `copykat`) — see [What's included](#whats-included)
- Both human (**hg20**) and mouse (**mm10**) genomes supported
- 17-patient 5-cancer benchmark vs R copykat: **median py↔R ARI 0.988**, **median wall-clock 23.4× faster** on 8 cores ([`benchmarks/full/FINDINGS.md`](benchmarks/full/FINDINGS.md))

> This is a **candidate standalone mirror** of the canonical implementation that will live in [`omicverse`](https://github.com/Starlitnightly/omicverse) (`omicverse.external.copykat_py`). Algorithmic work is developed here first and synced upstream for users who want CopyKAT without the full omicverse stack.

## Install

```bash
pip install pycopykat
```

or, from source with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Python 3.10–3.12 supported. Pure-Python wheel — no compiler required.

## Quick-start

```python
import pandas as pd
from pycopykat import copykat, CopykatConfig

counts = pd.read_csv("counts.tsv", sep="\t", index_col=0)  # genes × cells, raw ints

result = copykat(counts, config=CopykatConfig(sam_name="sample1", n_jobs=8))

result.prediction   # DataFrame[cell, copykat.pred] — aneuploid / diploid / not.defined
result.cna_mat      # bin × cell CNA matrix (MultiIndex: chrom, start, end)
result.linkage      # scipy Ward linkage matrix, (n-1, 4)
result.subclone     # Series: aneuploid cell → subclone label (int)
result.warnings     # runtime warnings (tuple[str, ...])
```

### AnnData batch driver

For a per-sample scanpy workflow, `copykat_by_batch` iterates over a batch column in a `.h5ad` and concatenates the per-sample predictions:

```python
import anndata as ad
from pycopykat import copykat_by_batch

adata = ad.read_h5ad("dataset.h5ad")
preds = copykat_by_batch(adata, batch_key="sample", n_jobs=8)
```

### CLI

```bash
pycopykat run-h5ad --input dataset.h5ad --batch-key sample --output out/ --n-jobs 8
```

### Mouse / mm10

```python
from pycopykat import copykat, CopykatConfig

result = copykat(
    mouse_counts,                                    # MGI symbols in rows
    config=CopykatConfig(genome="mm10", n_jobs=8),
)
```

Mirrors R copykat's mm10 code path: uses `mgi_symbol` as the annotation key and **skips** the HLA / cell-cycle gene filter (upstream behaviour).

## Low-level functional API (mirrors R one-to-one)

```python
from pycopykat.io.annotation import annotate_genes, annotate_gene_names
from pycopykat.cna.bins import aggregate_to_bins
from pycopykat.baseline.gmm import baseline_gmm
from pycopykat.baseline.auto import baseline_norm_cl
from pycopykat.segment.mcmc import segment_cells
from pycopykat.viz.heatmap import plot_cna_heatmap

# e.g. attach hg20 coordinates to a genes × cells frame
annotated = annotate_genes(counts, id_type="Symbol", genome="hg20")
```

## What's included

| Python | R counterpart | Purpose |
|---|---|---|
| `pycopykat.copykat` | `copykat` | Unsupervised end-to-end pipeline |
| `pycopykat.copykat_by_batch` | — | AnnData batch driver (pycopykat-only) |
| `pycopykat.CopykatConfig`, `CopykatResult` | — | `@dataclass` config / result containers |
| `io.annotation.annotate_genes`, `annotate_gene_names` | `annotateGenes.hg20`, `annotateGenes.mm10` | Attach gene coordinates; hg20 drops HLA + cycle, mm10 skips filters |
| `io.annotation.load_hg20_annotation` / `load_mm10_annotation` / `load_hg20_bins` | `full.anno` / `full.anno.mm10` / `DNA.hg20` | Parquet loaders for the reference tables |
| `cna.bins.aggregate_to_bins` | `convert.all.bins.hg20` | Gene-level CNA → 220 kb bin aggregation |
| `baseline.gmm.baseline_gmm` | `baseline.GMM` | GMM-based diploid pre-definition |
| `baseline.auto.baseline_norm_cl` | `baseline.norm.cl` | Integrative-clustering diploid baseline |
| `segment.mcmc.segment_cells`, `segment.breakpoint.find_breakpoints` | `CNA.MCMC` | KS breakpoints + MCMC segmentation |
| `classify.predict.predict_ploidy`, `classify.subclone.dynamic_tree_cut` | (inside `copykat`) | Per-cell prediction + subclone cutting |
| `viz.heatmap.plot_cna_heatmap` | `heatmap.3` | CNA heatmap rendering |
| `kernels.{pdist_euclidean, kalman_smooth, pg_posterior_mean, ...}` | internal Rcpp | Numba / BLAS hot-kernel implementations |
| `validation.{compare_predictions, compare_cna, run_r_copykat}` | — | R-parity harness (pycopykat-only) |

## Reproducing R results

pycopykat targets **empirical py↔R agreement on the same counts matrix under the unsupervised auto-baseline**, not bit-exact reproduction of every intermediate. Per-stage parity:

| Stage | Status vs R copykat |
|---|---|
| gene × cell preprocessing (filter + VST + log-Freeman–Tukey) | **bit-exact** on dense fixtures (≤ 1e-12) |
| 220 kb bin aggregation + DLM smoother (Kalman) | **bit-exact** up to float64 rounding (≤ 1e-10) |
| euclidean pdist (hot path) | ≤ 1e-8 absolute error vs `scipy.spatial.distance.pdist` |
| GMM baseline selection | **approximate** (sklearn GMM vs R `mixtools::normalmixEM`) |
| hierarchical clustering | **approximate** (scipy ward.D2 vs R `hclust(ward.D)`) |
| dynamic tree cut | **approximate** (`dynamicTreeCut` V1 port) |
| per-cell prediction | **empirical: median ARI 0.988 / κ 0.994** on 17-patient sweep |

Full mechanism breakdown of the 5 / 17 patients where py and R diverge: [`benchmarks/full/FINDINGS.md`](benchmarks/full/FINDINGS.md).

### Generating the R reference

```bash
Rscript tests/r_reference.R
uv run pytest tests/test_r_parity.py -v
```

`tests/r_reference.R` runs R copykat on the `exp.rawdata` fixture bundled with the upstream package and writes `tests/r_out/*.tsv`. The Python-side test skips cleanly when those TSVs are absent.

## Reproducing the 17-patient benchmark

The sweep in `benchmarks/full/` is driven by:

1. `scripts/run_all_benchmarks.py` — slices the five per-cancer counts matrices into per-patient `counts.tsv` + `cells.csv` and runs R copykat.
2. `scripts/run_py_sweep.py` — runs pycopykat on the same per-patient counts and invokes `scripts/compare_py_vs_r.py` to write per-patient `metrics.csv` + `figure_py_vs_R.png`.
3. `scripts/aggregate_py_vs_r_mechanism.py` — rolls the summary into a mechanism table (matched / R_label_flipped / partial / different_clusters).
4. `scripts/make_overview_figure.py` — renders the cross-cancer overview figure.

The `cells.csv` `umap1` / `umap2` columns ship from the 3CA release and are used as figure layout only — no external cell-type label is used by any metric.

## Testing

```bash
uv run pytest tests/unit tests/test_smoke.py          # fast unit + smoke tests (no R)
uv run pytest tests/test_r_parity.py                  # offline R-parity (skips without tests/r_out/)
uv run pytest tests/test_regression.py -m slow        # end-to-end vs R copykat (invokes R)
```

## Relationship to omicverse

Developed following the [omicverse-to-developer](https://github.com/omicverse/omicverse-to-developer) `py-<Name>` conventions (pure-Python, no `rpy2` in production code, AnnData-native I/O, Numba only on hot kernels). Upstream integration plan:

- Canonical implementation: `omicverse.external.copykat_py` (pending)
- Standalone mirror (this repo): same code, same API, without the full omicverse packaging

## Citation

If you use this package, please cite the original CopyKAT paper:

> Gao, R., Bai, S., Henderson, Y.C. *et al.* **Delineating copy number and clonal substructure in human tumors from single-cell transcriptomes.** *Nature Biotechnology* **39**, 599–608 (2021). <https://doi.org/10.1038/s41587-020-00795-2>

and acknowledge omicverse / this repository for the Python port.

## License

GPL-2.0-or-later — derivative of GPL-2.0 CopyKAT by Ruli Gao.
