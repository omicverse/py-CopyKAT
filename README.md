# pycopykat

Python rewrite of [CopyKAT](https://github.com/navinlabcode/copykat)
(Gao et al. *Nat Biotechnol* 2021) with NumPy / SciPy / Numba
acceleration. Derivative of GPL-2.0 CopyKAT by Ruli Gao; licensed
GPL-2.0-or-later.

pycopykat takes the same genes × cells integer counts matrix that R
copykat consumes and produces the same per-cell aneuploid / diploid
prediction and bin × cell CNA matrix under unsupervised auto-baseline
selection.

## Relation to omicverse

pycopykat is developed following the
[omicverse-to-developer](https://github.com/omicverse/omicverse-to-developer)
`py-<Name>` conventions (pure-Python, no `rpy2` in production code,
AnnData-native I/O, Numba only on hot kernels). It is a **candidate
standalone mirror** — once vendored into the
[`omicverse`](https://github.com/Starlitnightly/omicverse) organisation,
this repository will serve as the maintenance mirror of
`omicverse.external.copykat_py`, and users wanting CopyKAT without the
full omicverse stack can continue to `pip install pycopykat`. Algorithmic
work lives here first and will sync upstream.

## Benchmark headline

17-patient 5-cancer sweep vs R copykat (unsupervised auto-baseline,
same counts matrix, 8 cores):

* median **py↔R ARI 0.988**, mean 0.922
* median **wall-clock speedup 23.4×** (mean 31.3×, max 82.4×)
* per-patient metrics in `benchmarks/full/py_vs_r_summary.csv`;
  mechanism breakdown in `benchmarks/full/mechanism_summary.csv`;
  full write-up in [`benchmarks/full/FINDINGS.md`](benchmarks/full/FINDINGS.md).

## Installation

```bash
# with uv (recommended)
uv sync

# or with pip
pip install -e .
```

Python 3.10–3.12 supported.

## Usage

### Library API

```python
import pandas as pd
from pycopykat import copykat, CopykatConfig

counts = pd.read_csv("counts.tsv", sep="\t", index_col=0)  # genes × cells
result = copykat(counts, config=CopykatConfig(n_jobs=8, sam_name="sample"))

result.prediction       # DataFrame[cell, copykat.pred] with aneuploid/diploid/not.defined
result.cna_mat          # bin × cell CNA matrix (MultiIndex: chrom, start, end)
result.linkage          # scipy Ward linkage matrix, (n-1, 4)
result.subclone         # Series: aneuploid cell → subclone label (int)
result.warnings         # tuple of runtime warning strings
```

### h5ad batch driver

For a per-sample anndata workflow, `copykat_by_batch` iterates over a
batch column in a `.h5ad` and concatenates the per-sample predictions:

```python
import anndata as ad
from pycopykat import copykat_by_batch

adata = ad.read_h5ad("dataset.h5ad")
preds = copykat_by_batch(adata, batch_key="sample", n_jobs=8)
```

The same workflow is exposed from the command line:

```bash
uv run pycopykat run-h5ad --input dataset.h5ad --batch-key sample \
    --output out/ --n-jobs 8
```

## Reproducing the benchmark

The 17-patient sweep in `benchmarks/full/` is driven by:

1. `scripts/run_all_benchmarks.py` — slices the five per-cancer counts
   matrices into per-patient `counts.tsv` + `cells.csv` and runs R
   copykat.
2. `scripts/run_py_sweep.py` — runs pycopykat on the same per-patient
   counts and invokes `scripts/compare_py_vs_r.py` to write per-patient
   `metrics.csv` + `figure_py_vs_R.png`.
3. `scripts/aggregate_py_vs_r_mechanism.py` — rolls the summary into
   a mechanism table (matched / R_label_flipped / partial /
   different_clusters).
4. `scripts/make_overview_figure.py` — renders the cross-cancer
   overview figure.

The `cells.csv` `umap1` / `umap2` columns ship from the 3CA release and
are used as figure layout only — no external cell-type label is used
by any metric.

## Parity status

pycopykat's correctness target is *py ↔ R agreement on the same counts
matrix under unsupervised auto-baseline*, not bit-exact reproduction of
every intermediate. The 17-patient 5-cancer sweep above is the empirical
gate; the per-stage status is:

| Stage | Status vs R copykat | Notes |
|---|---|---|
| gene × cell preprocessing (filter + VST + log-Freeman–Tukey) | **bit-exact** on dense fixtures | sparse path produces identical dense output to ≤ 1e-12 |
| 220 kb bin aggregation + DLM smoother (Kalman) | **bit-exact** up to float64 rounding (≤ 1e-10) | Fortran-order input to the Numba kernel; identical state-space parameters |
| euclidean pdist (hot path) | ≤ 1e-8 absolute error vs `scipy.spatial.distance.pdist` | uses BLAS GEMM identity form for N ≥ 100; small-N routes to scipy |
| GMM baseline selection | **approximate** | sklearn `GaussianMixture` vs R `mixtools::normalmixEM`; different EM init and tie-breaking |
| hierarchical clustering | **approximate** | `scipy.cluster.hierarchy` ward.D2 vs R `hclust(method="ward.D")`; see Murtagh & Legendre (2014) |
| dynamic tree cut | **approximate** | `dynamicTreeCut` V1 port; degenerate single-branch trees may disagree |
| per-cell prediction | **empirical: median ARI 0.988 / κ 0.994** on 17-patient sweep | three patients show inverted aneuploid/diploid labels (high ARI, negative κ) driven by baseline-selection divergence |

See [`benchmarks/full/FINDINGS.md`](benchmarks/full/FINDINGS.md) for the
mechanism breakdown of the 5 / 17 patients where py and R diverge.

## Testing

```bash
uv run pytest tests/unit -x                  # fast unit tests
uv run pytest tests/test_regression.py -m slow -x  # end-to-end vs R copykat
```

The `-m slow` suite invokes R copykat on a small pilot patient and
compares ARI / κ against a pinned threshold.
