# pycopykat

Python rewrite of [CopyKAT](https://github.com/navinlabcode/copykat)
(Gao et al. *Nat Biotechnol* 2021) with NumPy / SciPy / Numba
acceleration. Derivative of GPL-2.0 CopyKAT by Ruli Gao; licensed
GPL-2.0-or-later.

pycopykat takes the same genes × cells integer counts matrix that R
copykat consumes and produces the same per-cell aneuploid / diploid
prediction and bin × cell CNA matrix under unsupervised auto-baseline
selection.

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
result.cna_mat          # bin × cell CNA matrix
result.clustering       # Ward linkage + cluster assignments
result.warnings         # any runtime warnings
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

1. `scripts/run_all_benchmarks.py` — slices the five 3CA counts
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

## Testing

```bash
uv run pytest tests/unit -x                  # fast unit tests
uv run pytest tests/test_regression.py -m slow -x  # end-to-end vs R copykat
```

The `-m slow` suite invokes R copykat on a small pilot patient and
compares ARI / κ against a pinned threshold.
