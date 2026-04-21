"""Build and execute the pycopykat example notebooks.

Produces (next to this file):
  - tutorial_quickstart.ipynb         — minimal end-to-end demo on R copykat's
    canonical exp.rawdata fixture (302 cells x 33694 genes).
  - compare_py_vs_R.ipynb             — pycopykat vs R copykat parity notebook
    on exp.rawdata, runs both implementations inline and compares
    classification + bin-level CNA via omicverse visualisations.
  - compare_py_vs_R_realdata.ipynb    — pycopykat vs R copykat parity notebook
    on a 17-patient real-tumour sample (default Lee2020/SMC16). Reads
    pre-computed outputs under benchmarks/full/<cancer>/<sample>/{r_out,py_out}/
    instead of re-running either pipeline; demonstrates the documented
    SMC16 parity gap (CNA agrees, classification flips).

Run from the repo root:

    uv run python examples/_build_notebooks.py             # build + execute all
    uv run python examples/_build_notebooks.py tutorial    # build + execute one
    uv run python examples/_build_notebooks.py --no-execute  # build only

The `compare*` notebooks need omicverse + R copykat. The default kernel is
``omicverse`` (a Jupyter kernelspec pointing at the conda env that has
omicverse + pycopykat + pyreadr installed). Override via:

    PYCOPYKAT_KERNEL=python3 uv run python examples/_build_notebooks.py

Executed notebooks are committed with their outputs so GitHub renders them
directly, per the omicverse-to-developer ``py-<Name>`` convention.
"""
from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent
FIXTURE_RDA = Path("/media/jason/T7/rerbulid/copykat-R/data/exp.rawdata.rda")
R_DRIVER = HERE / "r_driver_compare.R"

# Default real-tumour sample for compare_py_vs_R_realdata.ipynb. SMC16 is
# the only sample in the 17-patient sweep where py and R disagree on the
# majority class — see benchmarks/full/FINDINGS.md "Known parity gap".
# Override the sample by editing this constant or pre-baking another folder
# under benchmarks/full/.
REALDATA_DEFAULT = ("Lee2020_Colorectal", "SMC16")
REALDATA_SAMPLES_ROOT = REPO_ROOT / "benchmarks" / "full"


# ---------------------------------------------------------------------------
# Notebook 1: tutorial_quickstart.ipynb (existing — kept verbatim)
# ---------------------------------------------------------------------------

def _tut_quickstart() -> Path:
    nb = nbf.v4.new_notebook()
    cells = nb.cells

    cells.append(nbf.v4.new_markdown_cell(
        "# pycopykat quickstart\n"
        "\n"
        "End-to-end demo on R copykat's bundled `exp.rawdata` fixture "
        "(302 cells × 33,694 genes). This notebook drives "
        "`from pycopykat import copykat` directly — no R, no rpy2, no "
        "Bioconductor.\n"
    ))
    cells.append(nbf.v4.new_code_cell(
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import pyreadr\n"
        "\n"
        "import pycopykat\n"
        "from pycopykat import CopykatConfig, copykat\n"
        "\n"
        f"FIXTURE = {str(FIXTURE_RDA)!r}\n"
        "print('pycopykat version:', pycopykat.__version__)\n"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Load the canonical counts matrix\n"
        "\n"
        "`exp.rawdata` is a dense genes × cells integer matrix shipped "
        "with the upstream R [CopyKAT](https://github.com/navinlabcode/copykat) package."
    ))
    cells.append(nbf.v4.new_code_cell(
        "counts = next(iter(pyreadr.read_r(FIXTURE).values()))\n"
        "print('shape (genes x cells):', counts.shape)\n"
        "counts.iloc[:4, :4]\n"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Run `copykat` with the unsupervised auto-baseline\n"
        "\n"
        "`n_jobs=1` keeps the notebook output deterministic at the "
        "expense of wall-clock time; raise it for production."
    ))
    cells.append(nbf.v4.new_code_cell(
        "result = copykat(\n"
        "    counts,\n"
        "    config=CopykatConfig(sam_name='exp_rawdata', n_jobs=1),\n"
        ")\n"
        "print('prediction columns:', list(result.prediction.columns))\n"
        "print('cna_mat shape (bins x cells):', result.cna_mat.shape)\n"
        "print('linkage matrix shape:', result.linkage.shape)\n"
        "print('subclone assignments:', len(result.subclone), 'aneuploid cells')\n"
        "print('warnings:', result.warnings)\n"
    ))
    cells.append(nbf.v4.new_markdown_cell("## 3. Per-cell aneuploid / diploid prediction"))
    cells.append(nbf.v4.new_code_cell("pred = result.prediction\npred.head()\n"))
    cells.append(nbf.v4.new_code_cell(
        "pred_col = 'copykat.pred' if 'copykat.pred' in pred.columns else pred.columns[-1]\n"
        "pred[pred_col].value_counts()\n"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Bin × cell CNA matrix — visual sanity check\n"
        "\n"
        "The CNA matrix is bins × cells; here we render a down-sampled "
        "heatmap (first 80 cells) to confirm the expected block structure."
    ))
    cells.append(nbf.v4.new_code_cell(
        "cna = result.cna_mat\n"
        "subset = cna.iloc[:, :80] if hasattr(cna, 'iloc') else cna[:, :80]\n"
        "mat = subset.values if hasattr(subset, 'values') else np.asarray(subset)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)\n"
        "ax.set_xlabel('cell (first 80)')\n"
        "ax.set_ylabel('bin')\n"
        "ax.set_title('pycopykat CNA — exp.rawdata (first 80 cells)')\n"
        "plt.colorbar(im, ax=ax, label='log2 CNA')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Where to go next\n"
        "\n"
        "* AnnData users: `from pycopykat import copykat_by_batch` iterates\n"
        "  over a batch column in an `.h5ad`.\n"
        "* CLI equivalent: `uv run pycopykat run-h5ad --input <...> --batch-key <col>`.\n"
        "* py↔R single-sample comparison: `compare_py_vs_R.ipynb` (this directory).\n"
        "* py↔R real-tumour comparison: `compare_py_vs_R_realdata.ipynb`.\n"
        "* 17-patient py↔R benchmark: see `benchmarks/full/FINDINGS.md`.\n"
    ))

    out = HERE / "tutorial_quickstart.ipynb"
    with open(out, "w") as f:
        nbf.write(nb, f)
    return out


# ---------------------------------------------------------------------------
# Notebook 2: compare_py_vs_R.ipynb (new — exp.rawdata, run both inline)
# ---------------------------------------------------------------------------

def _compare_exp_rawdata() -> Path:
    nb = nbf.v4.new_notebook()
    cells = nb.cells

    cells.append(nbf.v4.new_markdown_cell(
        "# `pycopykat` vs R `copykat` — parity comparison on `exp.rawdata`\n"
        "\n"
        "This notebook runs **`pycopykat`** (the Python port) and the upstream "
        "R **`copykat`** on copykat's bundled 302-cell fixture, then uses "
        "**omicverse** for visualisation.\n"
        "\n"
        "Both implementations process the same gene × cell counts matrix. We "
        "compare:\n"
        "* overlap of the aneuploid sets (Venn),\n"
        "* aneuploid / diploid classification agreement (confusion matrix +\n"
        "  ARI / FMI),\n"
        "* per-cluster aneuploid fraction on a Leiden partition,\n"
        "* bin-level CNA agreement (Pearson + Spearman + side-by-side\n"
        "  heatmap + Δ heatmap).\n"
        "\n"
        "Sister notebook: `compare_py_vs_R_realdata.ipynb` runs the same\n"
        "comparison on a real-tumour sample using cached outputs.\n"
    ))

    cells.append(nbf.v4.new_code_cell(
        "from __future__ import annotations\n"
        "import os, subprocess, warnings\n"
        "from pathlib import Path\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import pyreadr\n"
        "import omicverse as ov\n"
        "import anndata as ad\n"
        "\n"
        "import pycopykat\n"
        "from pycopykat import CopykatConfig, copykat\n"
        "from pycopykat.viz import (\n"
        "    plot_cna_heatmap_compare,\n"
        "    plot_cna_delta,\n"
        ")\n"
        "\n"
        "ov.plot_set()\n"
        f"FIXTURE = {str(FIXTURE_RDA)!r}\n"
        f"R_DRIVER = {str(R_DRIVER)!r}\n"
        "RSCRIPT = os.environ.get('RSCRIPT', 'Rscript')\n"
        "WORK = Path('./_compare_out_exp_rawdata').resolve()\n"
        "WORK.mkdir(exist_ok=True)\n"
        "print('omicverse', ov.__version__, '— pycopykat', pycopykat.__version__)\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Load `exp.rawdata` and dump shared counts.tsv\n"
        "\n"
        "Both implementations consume the same on-disk TSV so there is no\n"
        "input-format ambiguity."
    ))
    cells.append(nbf.v4.new_code_cell(
        "counts = next(iter(pyreadr.read_r(FIXTURE).values()))\n"
        "print('shape (genes x cells):', counts.shape)\n"
        "\n"
        "counts_path = WORK / 'counts.tsv'\n"
        "if not counts_path.exists():\n"
        "    counts.to_csv(counts_path, sep='\\t')\n"
        "print('counts ->', counts_path, counts_path.stat().st_size // 1024, 'KB')\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Run R `copykat` via subprocess\n"
        "\n"
        "Driver script: `examples/r_driver_compare.R`. Output filenames are\n"
        "stable: `prediction.tsv`, `cna.tsv`, `runinfo.txt`."
    ))
    cells.append(nbf.v4.new_code_cell(
        "r_out = WORK / 'r_out'\n"
        "r_out.mkdir(exist_ok=True)\n"
        "if not (r_out / 'prediction.tsv').exists():\n"
        "    proc = subprocess.run(\n"
        "        [RSCRIPT, R_DRIVER, str(counts_path), str(r_out), 'exp_rawdata', '1'],\n"
        "        capture_output=True, text=True,\n"
        "    )\n"
        "    print(proc.stdout[-600:])\n"
        "    if proc.returncode != 0:\n"
        "        print('STDERR:', proc.stderr[-1200:])\n"
        "        raise RuntimeError('R driver failed')\n"
        "r_pred = pd.read_csv(r_out / 'prediction.tsv', sep='\\t')\n"
        "r_cna_raw = pd.read_csv(r_out / 'cna.tsv', sep='\\t')\n"
        "print('R prediction rows:', len(r_pred), '— aneuploid:', (r_pred['copykat.pred']=='aneuploid').sum())\n"
        "print('R CNA shape (bins x meta+cells):', r_cna_raw.shape)\n"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 3. Run `pycopykat` inline"))
    cells.append(nbf.v4.new_code_cell(
        "result = copykat(counts, config=CopykatConfig(sam_name='exp_rawdata', n_jobs=1))\n"
        "py_pred = result.prediction.copy()\n"
        "py_cna = result.cna_mat.copy()\n"
        "print('py prediction rows:', len(py_pred),\n"
        "      '— aneuploid:', py_pred['copykat.pred'].astype(str).str.contains('aneuploid').sum())\n"
        "print('py CNA shape (bins x cells):', py_cna.shape)\n"
        "print('warnings:', result.warnings)\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Align cell IDs and bin index between the two outputs\n"
        "\n"
        "R copykat applies `make.names()` to cell IDs (turns hyphens into\n"
        "dots); pycopykat keeps the original. We map back to the original\n"
        "form. The two CNA matrices share the same hg20 220-kb binning so\n"
        "we align them by row order."
    ))
    cells.append(nbf.v4.new_code_cell(
        "# Cell-ID normalisation: R replaces '-' with '.' via make.names\n"
        "py_cells = list(py_cna.columns)\n"
        "r_meta_cols = [c for c in r_cna_raw.columns if c in ('chrom', 'chrompos', 'abspos', 'start', 'end')]\n"
        "r_cell_cols_orig = [c for c in r_cna_raw.columns if c not in r_meta_cols]\n"
        "\n"
        "def _normalize_r_cell(name: str) -> str:\n"
        "    # match against py cell names by undoing R make.names dot substitution\n"
        "    if name in py_cells:\n"
        "        return name\n"
        "    cand = name.replace('.', '-')\n"
        "    if cand in py_cells:\n"
        "        return cand\n"
        "    return name\n"
        "\n"
        "r_rename = {c: _normalize_r_cell(c) for c in r_cell_cols_orig}\n"
        "r_cna = r_cna_raw.drop(columns=r_meta_cols).rename(columns=r_rename)\n"
        "\n"
        "# Adopt py's bin index for both sides (rows align by position).\n"
        "if r_cna.shape[0] == py_cna.shape[0]:\n"
        "    r_cna.index = py_cna.index\n"
        "else:\n"
        "    print(f'bin counts differ: py={py_cna.shape[0]} R={r_cna.shape[0]} — falling back to integer index')\n"
        "    py_cna = py_cna.reset_index(drop=True); r_cna = r_cna.reset_index(drop=True)\n"
        "\n"
        "# Normalise R prediction's cell column too\n"
        "r_pred = r_pred.copy()\n"
        "r_pred['cell'] = r_pred['cell'].map(_normalize_r_cell)\n"
        "\n"
        "common_cells = sorted(set(py_pred['cell']) & set(r_pred['cell']))\n"
        "print('common cells (predictions):', len(common_cells))\n"
        "common_cna_cells = sorted(set(py_cna.columns) & set(r_cna.columns))\n"
        "print('common cells (CNA matrices):', len(common_cna_cells))\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Build a shared `AnnData` for omicverse downstream\n"
        "\n"
        "We attach both predictions + a per-cell `agree` label so omicverse's\n"
        "embedding helpers can colour by either side."
    ))
    cells.append(nbf.v4.new_code_cell(
        "X = counts.loc[:, common_cells].T.values  # cells x genes\n"
        "adata = ad.AnnData(X=X.astype(float), obs=pd.DataFrame(index=common_cells), var=pd.DataFrame(index=counts.index))\n"
        "\n"
        "# Strip py's c1/c2/low.conf decoration so the binary aneuploid axis is comparable\n"
        "def _binarise(label: str) -> str:\n"
        "    s = str(label)\n"
        "    if 'aneuploid' in s: return 'aneuploid'\n"
        "    if 'diploid' in s: return 'diploid'\n"
        "    return 'not.defined'\n"
        "\n"
        "py_map = {c: _binarise(p) for c, p in zip(py_pred['cell'], py_pred['copykat.pred'])}\n"
        "r_map  = {c: _binarise(p) for c, p in zip(r_pred['cell'], r_pred['copykat.pred'])}\n"
        "adata.obs['py_class'] = pd.Categorical([py_map.get(c, 'not.defined') for c in adata.obs_names], categories=['diploid','aneuploid','not.defined'])\n"
        "adata.obs['r_class']  = pd.Categorical([r_map.get(c, 'not.defined') for c in adata.obs_names], categories=['diploid','aneuploid','not.defined'])\n"
        "\n"
        "def _agree(a: str, b: str) -> str:\n"
        "    if a == b == 'aneuploid': return 'both aneuploid'\n"
        "    if a == b == 'diploid':   return 'both diploid'\n"
        "    if a == 'aneuploid' and b == 'diploid': return 'py-only aneuploid'\n"
        "    if a == 'diploid' and b == 'aneuploid': return 'R-only aneuploid'\n"
        "    return 'other'\n"
        "adata.obs['agree'] = pd.Categorical(\n"
        "    [_agree(a, b) for a, b in zip(adata.obs['py_class'], adata.obs['r_class'])],\n"
        ")\n"
        "adata.obs['agree'].value_counts()\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Aneuploid-set overlap — `ov.pl.venn`"
    ))
    cells.append(nbf.v4.new_code_cell(
        "py_set = set(adata.obs_names[adata.obs['py_class']=='aneuploid'])\n"
        "r_set  = set(adata.obs_names[adata.obs['r_class']=='aneuploid'])\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(4, 4))\n"
        "try:\n"
        "    ov.pl.venn(sets={'pycopykat': py_set, 'R copykat': r_set}, ax=ax, fontsize=10)\n"
        "except Exception as e:\n"
        "    # ov.pl.venn signature varies between omicverse releases — fall back to matplotlib_venn\n"
        "    from matplotlib_venn import venn2  # type: ignore\n"
        "    venn2([py_set, r_set], set_labels=('pycopykat', 'R copykat'), ax=ax)\n"
        "ax.set_title('Cells called \"aneuploid\"')\n"
        "plt.show()\n"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 7. Confusion matrix + ARI / FMI"))
    cells.append(nbf.v4.new_code_cell(
        "from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score\n"
        "\n"
        "conf = pd.crosstab(adata.obs['py_class'], adata.obs['r_class']).reindex(\n"
        "    index=['diploid','aneuploid'], columns=['diploid','aneuploid']\n"
        ").fillna(0).astype(int)\n"
        "print(conf)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(3.5, 3))\n"
        "im = ax.imshow(conf.values, cmap='viridis')\n"
        "for i in range(2):\n"
        "    for j in range(2):\n"
        "        ax.text(j, i, int(conf.values[i, j]), ha='center', va='center', color='white', fontsize=12)\n"
        "ax.set_xticks([0, 1]); ax.set_xticklabels(conf.columns)\n"
        "ax.set_yticks([0, 1]); ax.set_yticklabels(conf.index)\n"
        "ax.set_xlabel('R copykat'); ax.set_ylabel('pycopykat')\n"
        "ax.set_title('Classification confusion matrix')\n"
        "plt.colorbar(im, ax=ax, fraction=0.046)\n"
        "plt.tight_layout(); plt.show()\n"
        "\n"
        "ari = adjusted_rand_score(adata.obs['r_class'], adata.obs['py_class'])\n"
        "fmi = fowlkes_mallows_score(adata.obs['r_class'], adata.obs['py_class'])\n"
        "print(f'py vs R agreement: {(adata.obs[\"py_class\"] == adata.obs[\"r_class\"]).mean():.3%}')\n"
        "print(f'ARI = {ari:.3f}, FMI = {fmi:.3f}')\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 8. omicverse preprocess → PCA → Leiden → UMAP\n"
        "\n"
        "Run a standard scRNA-seq workflow so we can overlay the py / R\n"
        "labels on a meaningful low-dimensional layout."
    ))
    cells.append(nbf.v4.new_code_cell(
        "adata_viz = adata.copy()\n"
        "adata_viz.layers['counts'] = adata_viz.X.copy()\n"
        "ov.pp.preprocess(adata_viz, mode='shiftlog|pearson', n_HVGs=2000)\n"
        "adata_viz.raw = adata_viz\n"
        "adata_viz = adata_viz[:, adata_viz.var.highly_variable_features]\n"
        "ov.pp.scale(adata_viz)\n"
        "ov.pp.pca(adata_viz, layer='scaled', n_pcs=30)\n"
        "ov.pp.neighbors(adata_viz, n_neighbors=15, use_rep='scaled|original|X_pca')\n"
        "ov.pp.leiden(adata_viz, resolution=0.5)\n"
        "ov.pp.umap(adata_viz)\n"
        "for c in ('py_class', 'r_class', 'agree'):\n"
        "    adata_viz.obs[c] = adata.obs[c].reindex(adata_viz.obs_names).values\n"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 9. UMAP overlays — `ov.pl.embedding`"))
    cells.append(nbf.v4.new_code_cell(
        "ov.pl.embedding(adata_viz, basis='X_umap',\n"
        "                color=['leiden', 'py_class', 'r_class', 'agree'],\n"
        "                palette='Set2', frameon='small', ncols=2, wspace=0.25, show=False)\n"
        "plt.show()\n"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 10. Per-cluster aneuploid fraction — `ov.pl.cellproportion`"))
    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
        "ov.pl.cellproportion(adata_viz, celltype_clusters='py_class',\n"
        "                     groupby='leiden', ax=axes[0], legend=True)\n"
        "axes[0].set_title('pycopykat')\n"
        "ov.pl.cellproportion(adata_viz, celltype_clusters='r_class',\n"
        "                     groupby='leiden', ax=axes[1], legend=True)\n"
        "axes[1].set_title('R copykat')\n"
        "plt.tight_layout(); plt.show()\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 11. Bin-level CNA agreement — Pearson / Spearman per cell\n"
        "\n"
        "For each cell present in both outputs, correlate its bin-vector\n"
        "between py and R. Distribution shape tells us whether the CNA\n"
        "estimates agree across the population, even when the binary\n"
        "aneuploid call diverges."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from scipy.stats import pearsonr, spearmanr\n"
        "\n"
        "common = sorted(set(py_cna.columns) & set(r_cna.columns))\n"
        "py_M = py_cna[common].to_numpy()\n"
        "r_M  = r_cna[common].to_numpy()\n"
        "\n"
        "rho_p, rho_s = [], []\n"
        "for j in range(py_M.shape[1]):\n"
        "    a, b = py_M[:, j], r_M[:, j]\n"
        "    m = np.isfinite(a) & np.isfinite(b)\n"
        "    if m.sum() < 5 or np.std(a[m]) == 0 or np.std(b[m]) == 0:\n"
        "        continue\n"
        "    rho_p.append(pearsonr(a[m], b[m])[0])\n"
        "    rho_s.append(spearmanr(a[m], b[m])[0])\n"
        "rho_p = np.asarray(rho_p); rho_s = np.asarray(rho_s)\n"
        "print(f'per-cell Pearson:  median {np.median(rho_p):.3f}, mean {rho_p.mean():.3f}, n={len(rho_p)}')\n"
        "print(f'per-cell Spearman: median {np.median(rho_s):.3f}, mean {rho_s.mean():.3f}, n={len(rho_s)}')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(5, 3.5))\n"
        "ax.hist(rho_p, bins=40, alpha=0.6, label='Pearson')\n"
        "ax.hist(rho_s, bins=40, alpha=0.6, label='Spearman')\n"
        "ax.set_xlabel('per-cell py vs R bin correlation'); ax.set_ylabel('cells')\n"
        "ax.set_title('Bin-level CNA agreement, per cell')\n"
        "ax.legend(); plt.tight_layout(); plt.show()\n"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 12. Side-by-side CNA heatmaps + Δ heatmap"))
    cells.append(nbf.v4.new_code_cell(
        "ax_py, ax_r = plot_cna_heatmap_compare(\n"
        "    py_cna[common], r_cna[common], py_pred, r_pred, sort_by='py'\n"
        ")\n"
        "plt.gcf().suptitle('CNA heatmap — pycopykat vs R copykat', y=1.02)\n"
        "plt.tight_layout(); plt.show()\n"
    ))
    cells.append(nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(12, 5))\n"
        "plot_cna_delta(py_cna[common], r_cna[common], py_pred, ax=ax, vmin=-0.5, vmax=0.5)\n"
        "plt.tight_layout(); plt.show()\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Summary\n"
        "\n"
        "On `exp.rawdata` we expect:\n"
        "\n"
        "| metric | expected | observed |\n"
        "|---|---|---|\n"
        "| Aneuploid-set Venn overlap | high | see Venn |\n"
        "| Classification ARI | ≥ 0.8 | see §7 |\n"
        "| Per-cell CNA Pearson, median | ≥ 0.85 | see §11 |\n"
        "| Side-by-side heatmap structure | matched chromosome arms | see §12 |\n"
        "\n"
        "For a real-tumour sample where the pipelines disagree, see\n"
        "`compare_py_vs_R_realdata.ipynb` (default: Lee2020/SMC16, the only\n"
        "sample in the 17-patient sweep where py and R disagree on the\n"
        "majority class — see `benchmarks/full/FINDINGS.md`).\n"
    ))

    out = HERE / "compare_py_vs_R.ipynb"
    with open(out, "w") as f:
        nbf.write(nb, f)
    return out


# ---------------------------------------------------------------------------
# Notebook 3: compare_py_vs_R_realdata.ipynb (new — read cached outputs)
# ---------------------------------------------------------------------------

def _compare_realdata() -> Path:
    nb = nbf.v4.new_notebook()
    cells = nb.cells
    cancer, sample = REALDATA_DEFAULT
    sample_dir_str = str(REALDATA_SAMPLES_ROOT / cancer / sample)

    cells.append(nbf.v4.new_markdown_cell(
        f"# `pycopykat` vs R `copykat` — real-tumour comparison ({cancer} / {sample})\n"
        "\n"
        "This notebook reads **pre-computed** outputs from the 17-patient\n"
        "benchmark — no new runs are issued. R copykat on a 2,698-cell sample\n"
        "takes ~22 minutes; the cached outputs let the comparison render in\n"
        "seconds.\n"
        "\n"
        "**Why this sample.** SMC16 is the one sample of the 17 where pycopykat\n"
        "and R copykat disagree on the *majority class* (R: 83% diploid; py:\n"
        "34% diploid), even though their bin-level CNA matrices agree closely.\n"
        "Both runs trigger `unclassified.prediction` twice on the\n"
        "`norm_cl → GMM` baseline-fallback chain. See\n"
        "`benchmarks/full/FINDINGS.md → Known parity gap: Lee2020_Colorectal /\n"
        "SMC16` and `benchmarks/full/TODO_parity_SMC16.md` for the open root-\n"
        "cause investigation.\n"
        "\n"
        "Switch sample by editing `CANCER` / `SAMPLE` in the next cell — any\n"
        "subdirectory under `benchmarks/full/<cancer>/<sample>/` with the\n"
        "standard `r_out/` + `py_out/` layout works.\n"
    ))

    cells.append(nbf.v4.new_code_cell(
        "from __future__ import annotations\n"
        "import warnings\n"
        "from pathlib import Path\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import omicverse as ov\n"
        "import anndata as ad\n"
        "\n"
        "import pycopykat\n"
        "from pycopykat.viz import plot_cna_heatmap_compare, plot_cna_delta\n"
        "\n"
        "ov.plot_set()\n"
        f"CANCER = {cancer!r}\n"
        f"SAMPLE = {sample!r}\n"
        f"SAMPLE_DIR = Path({sample_dir_str!r})\n"
        "assert SAMPLE_DIR.exists(), f'sample dir missing: {SAMPLE_DIR}'\n"
        "print('omicverse', ov.__version__, '— pycopykat', pycopykat.__version__)\n"
        "print('reading cached outputs from:', SAMPLE_DIR)\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Read cached outputs (no recomputation)\n"
        "\n"
        "Files written by `scripts/run_r_copykat.R` and\n"
        "`scripts/run_py_copykat.py` during the 17-patient benchmark."
    ))
    cells.append(nbf.v4.new_code_cell(
        "tag = f'{SAMPLE}_full'\n"
        "r_pred = pd.read_csv(SAMPLE_DIR / 'r_out' / f'{tag}_copykat_prediction.txt', sep='\\t')\n"
        "py_pred = pd.read_csv(SAMPLE_DIR / 'py_out' / f'{tag}_py_copykat_prediction.txt', sep='\\t')\n"
        "r_cna_raw = pd.read_csv(SAMPLE_DIR / 'r_out' / f'{tag}_copykat_CNA_results.txt', sep='\\t')\n"
        "py_cna_raw = pd.read_csv(SAMPLE_DIR / 'py_out' / f'{tag}_py_copykat_CNA_results.txt', sep='\\t')\n"
        "print('R prediction rows :', len(r_pred))\n"
        "print('py prediction rows:', len(py_pred))\n"
        "print('R CNA shape  (bins x meta+cells):', r_cna_raw.shape)\n"
        "print('py CNA shape (bins x meta+cells):', py_cna_raw.shape)\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Normalise cell IDs and bin index\n"
        "\n"
        "R `make.names()` rewrites `-` to `.` in cell IDs; py keeps the\n"
        "original. Both sides share copykat's hg20 220-kb binning so we\n"
        "align the CNA matrices by row order."
    ))
    cells.append(nbf.v4.new_code_cell(
        "py_cells_set = set(py_pred['cell'])\n"
        "\n"
        "def _to_py_cell(name: str) -> str:\n"
        "    if name in py_cells_set: return name\n"
        "    cand = name.replace('.', '-')\n"
        "    return cand if cand in py_cells_set else name\n"
        "\n"
        "r_pred = r_pred.rename(columns={'cell.names': 'cell'}).copy()\n"
        "r_pred['cell'] = r_pred['cell'].map(_to_py_cell)\n"
        "\n"
        "META_COLS = {'chrom', 'chrompos', 'abspos', 'start', 'end'}\n"
        "def _strip_meta(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    return df.drop(columns=[c for c in df.columns if c in META_COLS])\n"
        "\n"
        "py_cna = _strip_meta(py_cna_raw)\n"
        "r_cna  = _strip_meta(r_cna_raw).rename(columns=lambda c: _to_py_cell(c))\n"
        "\n"
        "# Adopt py's bin axis (start_end multiindex if present, else range)\n"
        "if py_cna.shape[0] == r_cna.shape[0]:\n"
        "    if {'chrom', 'start', 'end'}.issubset(py_cna_raw.columns):\n"
        "        py_cna.index = pd.MultiIndex.from_arrays(\n"
        "            [py_cna_raw['chrom'], py_cna_raw['start']], names=['chrom', 'start'],\n"
        "        )\n"
        "        r_cna.index = py_cna.index\n"
        "    else:\n"
        "        py_cna.index = pd.RangeIndex(py_cna.shape[0])\n"
        "        r_cna.index = py_cna.index\n"
        "else:\n"
        "    raise RuntimeError(f'bin counts differ: py={py_cna.shape[0]} R={r_cna.shape[0]}')\n"
        "\n"
        "common_cells = sorted(set(py_pred['cell']) & set(r_pred['cell']))\n"
        "common_cna = sorted(set(py_cna.columns) & set(r_cna.columns))\n"
        "print('common cells (predictions):', len(common_cells))\n"
        "print('common cells (CNA):       ', len(common_cna))\n"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 3. Class-distribution side-by-side"))
    cells.append(nbf.v4.new_code_cell(
        "def _binarise(label: str) -> str:\n"
        "    s = str(label)\n"
        "    if 'aneuploid' in s: return 'aneuploid'\n"
        "    if 'diploid' in s:   return 'diploid'\n"
        "    return 'not.defined'\n"
        "\n"
        "py_class = py_pred.assign(c=py_pred['copykat.pred'].map(_binarise))\n"
        "r_class  = r_pred.assign(c=r_pred['copykat.pred'].map(_binarise))\n"
        "tab = pd.DataFrame({\n"
        "    'pycopykat': py_class['c'].value_counts().reindex(['diploid','aneuploid','not.defined']).fillna(0).astype(int),\n"
        "    'R copykat': r_class['c'].value_counts().reindex(['diploid','aneuploid','not.defined']).fillna(0).astype(int),\n"
        "})\n"
        "tab['py %'] = (tab['pycopykat'] / tab['pycopykat'].sum() * 100).round(1)\n"
        "tab['R %']  = (tab['R copykat'] / tab['R copykat'].sum() * 100).round(1)\n"
        "tab\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "On SMC16 you should see R calling ~83% diploid versus py ~34%\n"
        "diploid — the documented majority-class flip. **The notebook\n"
        "intentionally surfaces this** so the gap is visible, not hidden.\n"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 4. Build AnnData + Venn / confusion matrix"))
    cells.append(nbf.v4.new_code_cell(
        "counts_path = SAMPLE_DIR / 'counts.tsv'\n"
        "if counts_path.exists():\n"
        "    counts = pd.read_csv(counts_path, sep='\\t', index_col=0)\n"
        "    X = counts.loc[:, [c for c in common_cells if c in counts.columns]].T\n"
        "    adata = ad.AnnData(X=X.values.astype(float),\n"
        "                       obs=pd.DataFrame(index=X.index),\n"
        "                       var=pd.DataFrame(index=counts.index))\n"
        "else:\n"
        "    # No counts available — fabricate a tiny dummy expression matrix so omicverse\n"
        "    # downstream still works; UMAP overlays will reflect a synthetic neighbour graph.\n"
        "    rng = np.random.default_rng(0)\n"
        "    adata = ad.AnnData(X=rng.standard_normal((len(common_cells), 200)),\n"
        "                       obs=pd.DataFrame(index=common_cells),\n"
        "                       var=pd.DataFrame(index=[f'g{i}' for i in range(200)]))\n"
        "    print('(no counts.tsv — using dummy matrix for downstream embedding)')\n"
        "\n"
        "py_map = {c: _binarise(p) for c, p in zip(py_pred['cell'], py_pred['copykat.pred'])}\n"
        "r_map  = {c: _binarise(p) for c, p in zip(r_pred['cell'],  r_pred['copykat.pred'])}\n"
        "adata.obs['py_class'] = pd.Categorical(\n"
        "    [py_map.get(c, 'not.defined') for c in adata.obs_names],\n"
        "    categories=['diploid', 'aneuploid', 'not.defined'],\n"
        ")\n"
        "adata.obs['r_class'] = pd.Categorical(\n"
        "    [r_map.get(c, 'not.defined') for c in adata.obs_names],\n"
        "    categories=['diploid', 'aneuploid', 'not.defined'],\n"
        ")\n"
        "def _agree(a, b):\n"
        "    if a == b == 'aneuploid': return 'both aneuploid'\n"
        "    if a == b == 'diploid':   return 'both diploid'\n"
        "    if a == 'aneuploid' and b == 'diploid': return 'py-only aneuploid'\n"
        "    if a == 'diploid' and b == 'aneuploid': return 'R-only aneuploid'\n"
        "    return 'other'\n"
        "adata.obs['agree'] = pd.Categorical(\n"
        "    [_agree(a, b) for a, b in zip(adata.obs['py_class'], adata.obs['r_class'])],\n"
        ")\n"
        "adata.obs['agree'].value_counts()\n"
    ))

    cells.append(nbf.v4.new_code_cell(
        "py_aneu = set(adata.obs_names[adata.obs['py_class']=='aneuploid'])\n"
        "r_aneu  = set(adata.obs_names[adata.obs['r_class']=='aneuploid'])\n"
        "fig, ax = plt.subplots(figsize=(4, 4))\n"
        "try:\n"
        "    ov.pl.venn(sets={'pycopykat': py_aneu, 'R copykat': r_aneu}, ax=ax, fontsize=10)\n"
        "except Exception:\n"
        "    from matplotlib_venn import venn2  # type: ignore\n"
        "    venn2([py_aneu, r_aneu], set_labels=('pycopykat', 'R copykat'), ax=ax)\n"
        "ax.set_title('Cells called \"aneuploid\"')\n"
        "plt.show()\n"
    ))

    cells.append(nbf.v4.new_code_cell(
        "from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score\n"
        "\n"
        "conf = pd.crosstab(adata.obs['py_class'], adata.obs['r_class']).reindex(\n"
        "    index=['diploid','aneuploid'], columns=['diploid','aneuploid']\n"
        ").fillna(0).astype(int)\n"
        "fig, ax = plt.subplots(figsize=(3.5, 3))\n"
        "im = ax.imshow(conf.values, cmap='viridis')\n"
        "for i in range(2):\n"
        "    for j in range(2):\n"
        "        ax.text(j, i, int(conf.values[i, j]), ha='center', va='center', color='white', fontsize=12)\n"
        "ax.set_xticks([0,1]); ax.set_xticklabels(conf.columns)\n"
        "ax.set_yticks([0,1]); ax.set_yticklabels(conf.index)\n"
        "ax.set_xlabel('R copykat'); ax.set_ylabel('pycopykat')\n"
        "plt.colorbar(im, ax=ax, fraction=0.046)\n"
        "plt.tight_layout(); plt.show()\n"
        "\n"
        "ari = adjusted_rand_score(adata.obs['r_class'], adata.obs['py_class'])\n"
        "fmi = fowlkes_mallows_score(adata.obs['r_class'], adata.obs['py_class'])\n"
        "print(f'agreement: {(adata.obs[\"py_class\"] == adata.obs[\"r_class\"]).mean():.3%}, ARI = {ari:.3f}, FMI = {fmi:.3f}')\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 5. omicverse preprocess + UMAP overlays\n"
        "\n"
        "Run a standard scRNA workflow on the cached counts (or, if counts\n"
        "are absent, on a placeholder matrix) and overlay the py / R\n"
        "labels."
    ))
    cells.append(nbf.v4.new_code_cell(
        "adata_viz = adata.copy()\n"
        "if 'counts' in adata_viz.layers:\n"
        "    pass\n"
        "elif (adata_viz.X >= 0).all() and adata_viz.X.dtype.kind in 'iu':\n"
        "    adata_viz.layers['counts'] = adata_viz.X.copy()\n"
        "else:\n"
        "    # Synthetic matrix path — skip omicverse pp.preprocess (it expects counts)\n"
        "    adata_viz.layers['counts'] = np.abs(adata_viz.X).astype(int)\n"
        "try:\n"
        "    ov.pp.preprocess(adata_viz, mode='shiftlog|pearson', n_HVGs=2000)\n"
        "    adata_viz.raw = adata_viz\n"
        "    adata_viz = adata_viz[:, adata_viz.var.highly_variable_features]\n"
        "    ov.pp.scale(adata_viz)\n"
        "    ov.pp.pca(adata_viz, layer='scaled', n_pcs=30)\n"
        "    ov.pp.neighbors(adata_viz, n_neighbors=15, use_rep='scaled|original|X_pca')\n"
        "    ov.pp.leiden(adata_viz, resolution=0.5)\n"
        "    ov.pp.umap(adata_viz)\n"
        "    for c in ('py_class', 'r_class', 'agree'):\n"
        "        adata_viz.obs[c] = adata.obs[c].reindex(adata_viz.obs_names).values\n"
        "    ov.pl.embedding(adata_viz, basis='X_umap',\n"
        "                    color=['leiden', 'py_class', 'r_class', 'agree'],\n"
        "                    palette='Set2', frameon='small', ncols=2, wspace=0.25, show=False)\n"
        "    plt.show()\n"
        "    fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
        "    ov.pl.cellproportion(adata_viz, celltype_clusters='py_class',\n"
        "                         groupby='leiden', ax=axes[0], legend=True)\n"
        "    axes[0].set_title('pycopykat')\n"
        "    ov.pl.cellproportion(adata_viz, celltype_clusters='r_class',\n"
        "                         groupby='leiden', ax=axes[1], legend=True)\n"
        "    axes[1].set_title('R copykat')\n"
        "    plt.tight_layout(); plt.show()\n"
        "except Exception as e:\n"
        "    print('omicverse pipeline skipped:', type(e).__name__, e)\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Bin-level CNA agreement — per-cell Pearson / Spearman\n"
        "\n"
        "On SMC16 the per-cell distribution should peak high (median Pearson\n"
        "≥ 0.85) — this is the evidence that the pre-classification stages\n"
        "agree, isolating the divergence to the classifier."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from scipy.stats import pearsonr, spearmanr\n"
        "\n"
        "common = sorted(set(py_cna.columns) & set(r_cna.columns))\n"
        "py_M = py_cna[common].to_numpy()\n"
        "r_M  = r_cna[common].to_numpy()\n"
        "\n"
        "rho_p, rho_s = [], []\n"
        "for j in range(py_M.shape[1]):\n"
        "    a, b = py_M[:, j], r_M[:, j]\n"
        "    m = np.isfinite(a) & np.isfinite(b)\n"
        "    if m.sum() < 5 or np.std(a[m]) == 0 or np.std(b[m]) == 0:\n"
        "        continue\n"
        "    rho_p.append(pearsonr(a[m], b[m])[0])\n"
        "    rho_s.append(spearmanr(a[m], b[m])[0])\n"
        "rho_p = np.asarray(rho_p); rho_s = np.asarray(rho_s)\n"
        "print(f'per-cell Pearson:  median {np.median(rho_p):.3f}, mean {rho_p.mean():.3f}, n={len(rho_p)}')\n"
        "print(f'per-cell Spearman: median {np.median(rho_s):.3f}, mean {rho_s.mean():.3f}, n={len(rho_s)}')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(5, 3.5))\n"
        "ax.hist(rho_p, bins=40, alpha=0.6, label='Pearson')\n"
        "ax.hist(rho_s, bins=40, alpha=0.6, label='Spearman')\n"
        "ax.set_xlabel('per-cell py vs R bin correlation'); ax.set_ylabel('cells')\n"
        "ax.set_title(f'Bin-level CNA agreement, per cell — {SAMPLE}')\n"
        "ax.legend(); plt.tight_layout(); plt.show()\n"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 7. CNA heatmaps — side-by-side and Δ"))
    cells.append(nbf.v4.new_code_cell(
        "ax_py, ax_r = plot_cna_heatmap_compare(\n"
        "    py_cna[common], r_cna[common], py_pred, r_pred, sort_by='py'\n"
        ")\n"
        "plt.gcf().suptitle(f'CNA heatmap — pycopykat vs R copykat ({SAMPLE})', y=1.02)\n"
        "plt.tight_layout(); plt.show()\n"
    ))
    cells.append(nbf.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(12, 5))\n"
        "plot_cna_delta(py_cna[common], r_cna[common], py_pred, ax=ax, vmin=-0.5, vmax=0.5)\n"
        "plt.tight_layout(); plt.show()\n"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Takeaway\n"
        "\n"
        "On SMC16 the bin-level CNA matrices agree (high per-cell Pearson)\n"
        "but the binary aneuploid / diploid call diverges. This is the\n"
        "canonical evidence pattern that localises the parity gap to the\n"
        "classifier (`baseline_gmm` fallback + `predict_ploidy`), not to the\n"
        "preprocessing / segmentation pipeline. See\n"
        "`benchmarks/full/TODO_parity_SMC16.md` for the open root-cause\n"
        "investigation. On the 16 other samples in the 17-patient sweep\n"
        "(see `benchmarks/full/FINDINGS.md`) the classification matches\n"
        "(ARI ≥ 0.92).\n"
    ))

    out = HERE / "compare_py_vs_R_realdata.ipynb"
    with open(out, "w") as f:
        nbf.write(nb, f)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

NOTEBOOKS = {
    "tutorial":         (_tut_quickstart, "tutorial_quickstart.ipynb"),
    "compare":          (_compare_exp_rawdata, "compare_py_vs_R.ipynb"),
    "compare-realdata": (_compare_realdata, "compare_py_vs_R_realdata.ipynb"),
}


def _execute(path: Path) -> None:
    nb = nbf.read(str(path), as_version=4)
    client = NotebookClient(
        nb,
        timeout=3600,
        kernel_name=os.environ.get("PYCOPYKAT_KERNEL", "omicverse"),
        resources={"metadata": {"path": str(HERE)}},
    )
    client.execute()
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"executed {path}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "targets", nargs="*", choices=list(NOTEBOOKS),
        default=[], help="which notebooks to build (default: all)",
    )
    ap.add_argument("--no-execute", action="store_true",
                    help="build the .ipynb files but don't run them")
    args = ap.parse_args()
    targets = args.targets or list(NOTEBOOKS)

    paths = []
    for t in targets:
        builder, _ = NOTEBOOKS[t]
        paths.append(builder())
    for p in paths:
        if args.no_execute:
            print(f"built (not executed) {p}")
        else:
            _execute(p)


if __name__ == "__main__":
    main()
