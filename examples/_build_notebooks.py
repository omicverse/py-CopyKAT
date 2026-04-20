"""Build and execute the pycopykat example notebooks.

Produces (next to this file):
  - tutorial_quickstart.ipynb — minimal end-to-end demo on R copykat's
    canonical exp.rawdata fixture (302 cells x 33694 genes).

Run from the repo root:

    uv run python examples/_build_notebooks.py

The executed notebook is committed with its outputs so GitHub renders
it directly, per the omicverse-to-developer `py-<Name>` convention.
"""
from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent
FIXTURE_RDA = Path("/media/jason/T7/rerbulid/copykat-R/data/exp.rawdata.rda")


def _tut_quickstart() -> Path:
    nb = nbf.v4.new_notebook()
    cells = nb.cells

    cells.append(
        nbf.v4.new_markdown_cell(
            "# pycopykat quickstart\n"
            "\n"
            "End-to-end demo on R copykat's bundled `exp.rawdata` fixture "
            "(302 cells × 33,694 genes). This notebook drives "
            "`from pycopykat import copykat` directly — no R, no rpy2, no "
            "Bioconductor.\n"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
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
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. Load the canonical counts matrix\n"
            "\n"
            "`exp.rawdata` is a dense genes × cells integer matrix shipped "
            "with the upstream R [CopyKAT](https://github.com/navinlabcode/copykat) package."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "counts = next(iter(pyreadr.read_r(FIXTURE).values()))\n"
            "print('shape (genes x cells):', counts.shape)\n"
            "counts.iloc[:4, :4]\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. Run `copykat` with the unsupervised auto-baseline\n"
            "\n"
            "`n_jobs=1` keeps the notebook output deterministic at the "
            "expense of wall-clock time; raise it for production."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "result = copykat(\n"
            "    counts,\n"
            "    config=CopykatConfig(sam_name='exp_rawdata', n_jobs=1),\n"
            ")\n"
            "print('prediction columns:', list(result.prediction.columns))\n"
            "print('cna_mat shape (bins x cells):', result.cna_mat.shape)\n"
            "print('linkage matrix shape:', result.linkage.shape)\n"
            "print('subclone assignments:', len(result.subclone), 'aneuploid cells')\n"
            "print('warnings:', result.warnings)\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. Per-cell aneuploid / diploid prediction"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "pred = result.prediction\n"
            "pred.head()\n"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "pred_col = 'copykat.pred' if 'copykat.pred' in pred.columns else pred.columns[-1]\n"
            "pred[pred_col].value_counts()\n"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. Bin × cell CNA matrix — visual sanity check\n"
            "\n"
            "The CNA matrix is bins × cells; here we render a down-sampled "
            "heatmap (first 80 cells) to confirm the expected block structure."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
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
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. Where to go next\n"
            "\n"
            "* AnnData users: `from pycopykat import copykat_by_batch` iterates\n"
            "  over a batch column in an `.h5ad`.\n"
            "* CLI equivalent: `uv run pycopykat run-h5ad --input <...> --batch-key <col>`.\n"
            "* 17-patient py↔R benchmark: see `benchmarks/full/FINDINGS.md`.\n"
        )
    )

    out = HERE / "tutorial_quickstart.ipynb"
    with open(out, "w") as f:
        nbf.write(nb, f)
    return out


def _execute(path: Path) -> None:
    nb = nbf.read(str(path), as_version=4)
    client = NotebookClient(
        nb,
        timeout=1800,
        # Fall back to the default `python3` kernel — users may not have
        # a pycopykat-specific ipykernel registered.
        kernel_name=os.environ.get("PYCOPYKAT_KERNEL", "python3"),
        resources={"metadata": {"path": str(HERE)}},
    )
    client.execute()
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"executed {path}")


if __name__ == "__main__":
    nb_path = _tut_quickstart()
    _execute(nb_path)
