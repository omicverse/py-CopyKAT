# pycopykat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the R package CopyKAT in Python with Numba acceleration, matching R-version outputs to statistical equivalence (ARI ≥ 0.90, Spearman r ≥ 0.95) with ≥ 10× single-thread speedup on ~10k cell datasets.

**Architecture:** Layered pipeline (preprocess → baseline → segment → cna → classify → viz) with Numba-JIT hot kernels isolated in `kernels/`. Top-level `copykat()` API and a `pycopykat` CLI. Validation harness uses `subprocess` to call the installed R copykat 1.1.0 for reference generation.

**Tech Stack:** Python ≥3.10 <3.13, NumPy, SciPy, pandas, scikit-learn, Numba 0.59+, matplotlib, AnnData, pyreadr, typer. Build with `uv` + `hatchling`. Test with `pytest` + `pytest-benchmark` + `hypothesis`. Lint with `ruff` + `mypy`.

**Spec:** `docs/superpowers/specs/2026-04-19-pycopykat-design.md`

**R Reference:** `/media/jason/T7/rerbulid/copykat-R` (installed at `/home/jason/R/x86_64-pc-linux-gnu-library/4.5/copykat` v1.1.0, R 4.5.2)

**Repository root:** `/media/jason/T7/rerbulid/pycopykat`

---

## Milestone Overview

| M | Theme | Tasks |
|---|---|---|
| M1 | Project skeleton + hg20 reference data | 1.1 – 1.4 |
| M2 | Numba kernels (distances, Kalman, MCMC, adjust) | 2.1 – 2.5 |
| M3 | Pipeline front end (filter / annotate / normalize / smooth) | 3.1 – 3.5 |
| M4 | Baseline estimation (auto / GMM / synthetic) | 4.1 – 4.4 |
| M5 | Segmentation + bin conversion | 5.1 – 5.4 |
| M6 | Classification + subclone + heatmap | 6.1 – 6.4 |
| M7 | Pipeline orchestration + CLI + regression | 7.1 – 7.5 |
| M8 | 3CA external validation | 8.1 – 8.3 |
| M9 | Performance benchmark + release | 9.1 – 9.2 |

---

# M1 — Project Skeleton & Reference Data

## Task 1.1: Initialize project skeleton (pyproject + dirs)

**Files:**
- Create: `pyproject.toml`
- Create: `src/pycopykat/__init__.py`
- Create: `src/pycopykat/py.typed`
- Create: `LICENSE` (GPL-2.0 full text)
- Create: `README.md`
- Create: all package subdirectories with empty `__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

Contents:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pycopykat"
version = "0.1.0.dev0"
description = "Python rewrite of CopyKAT (Gao et al. 2021) with Numba acceleration"
readme = "README.md"
license = "GPL-2.0-or-later"
requires-python = ">=3.10,<3.13"
authors = [{name = "Jason"}]
dependencies = [
  "numpy>=1.26,<3",
  "scipy>=1.11",
  "pandas>=2.1",
  "scikit-learn>=1.4",
  "numba>=0.59",
  "matplotlib>=3.8",
  "anndata>=0.10",
  "pyreadr>=0.5",
  "typer>=0.9",
  "pyarrow>=14",
]

[project.optional-dependencies]
dev = [
  "pytest>=8",
  "pytest-benchmark>=4",
  "hypothesis>=6",
  "ruff>=0.4",
  "mypy>=1.9",
  "ipykernel",
]

[project.scripts]
pycopykat = "pycopykat.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/pycopykat"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "NPY"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"
```

- [ ] **Step 2: Create directory tree with empty `__init__.py`**

Run:
```bash
cd /media/jason/T7/rerbulid/pycopykat
mkdir -p src/pycopykat/{io,preprocess,baseline,segment,cna,classify,viz,kernels,validation}
mkdir -p tests/{data,unit}
mkdir -p data
mkdir -p scripts
for d in src/pycopykat src/pycopykat/io src/pycopykat/preprocess src/pycopykat/baseline \
         src/pycopykat/segment src/pycopykat/cna src/pycopykat/classify src/pycopykat/viz \
         src/pycopykat/kernels src/pycopykat/validation; do
  : > "$d/__init__.py"
done
touch src/pycopykat/py.typed
```

- [ ] **Step 3: Write `LICENSE`**

Copy GPL-2.0 text from https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt (already downloaded locally or from R copykat-R/Copyright). Ensure full text.

- [ ] **Step 4: Write minimal `README.md`**

Contents:
```markdown
# pycopykat

Python rewrite of [CopyKAT](https://github.com/navinlabcode/copykat) (Gao et al. *Nat Biotechnol* 2021) with Numba acceleration.

Derivative of GPL-2.0 CopyKAT by Ruli Gao. Licensed GPL-2.0-or-later.

## Status

V1 in development. See `docs/superpowers/specs/` and `docs/superpowers/plans/`.
```

- [ ] **Step 5: Verify env sync**

Run:
```bash
cd /media/jason/T7/rerbulid/pycopykat
uv venv --python 3.11
uv pip install -e ".[dev]"
uv run python -c "import pycopykat; print(pycopykat.__name__)"
```
Expected: `pycopykat` printed; no errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml LICENSE README.md src tests data scripts
git commit -m "chore: initialize pycopykat project skeleton"
```

---

## Task 1.2: Convert `sysdata.rda` to parquet reference data

R's `sysdata.rda` bundles the hg20 gene annotation table (`full.anno`), the 220KB bin table, and the cell-cycle gene list. We export these to parquet for fast Python load.

**Files:**
- Create: `scripts/convert_sysdata.R`
- Create: `data/hg20_gene_anno.parquet`
- Create: `data/hg20_220kb_bins.parquet`
- Create: `data/hg20_cycle_genes.txt`
- Create: `tests/unit/test_reference_data.py`

- [ ] **Step 1: Inspect sysdata.rda contents**

Run:
```bash
Rscript -e 'load("/media/jason/T7/rerbulid/copykat-R/data/sysdata.rda"); ls()'
```
Expected output lists objects like `full.anno`, `cyclegenes`, `DNA.hg20`, `full.annomm10`, etc. Note exact names.

- [ ] **Step 2: Write R conversion script**

Create `scripts/convert_sysdata.R`:
```r
#!/usr/bin/env Rscript
# Convert copykat sysdata.rda → parquet for pycopykat
suppressPackageStartupMessages({
  library(arrow)
})
load("/media/jason/T7/rerbulid/copykat-R/data/sysdata.rda")

# hg20 gene annotation
stopifnot(exists("full.anno"))
arrow::write_parquet(as.data.frame(full.anno),
                     "data/hg20_gene_anno.parquet")

# 220KB bin table
stopifnot(exists("DNA.hg20"))
arrow::write_parquet(as.data.frame(DNA.hg20),
                     "data/hg20_220kb_bins.parquet")

# Cell-cycle genes
stopifnot(exists("cyclegenes"))
writeLines(as.character(cyclegenes[[1]]),
           "data/hg20_cycle_genes.txt")

cat("Wrote:\n",
    "  data/hg20_gene_anno.parquet (", nrow(full.anno), " rows)\n",
    "  data/hg20_220kb_bins.parquet (", nrow(DNA.hg20), " rows)\n",
    "  data/hg20_cycle_genes.txt (", length(cyclegenes[[1]]), " genes)\n",
    sep = "")
```

- [ ] **Step 3: Ensure `arrow` R package is available**

Run:
```bash
Rscript -e 'if (!requireNamespace("arrow", quietly=TRUE)) install.packages("arrow", repos="https://cloud.r-project.org")'
```

- [ ] **Step 4: Write failing test that parquet files exist & shape sensible**

Create `tests/unit/test_reference_data.py`:
```python
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

def test_hg20_gene_anno_parquet_exists():
    df = pd.read_parquet(DATA / "hg20_gene_anno.parquet")
    assert len(df) > 20_000
    for col in ("hgnc_symbol", "chromosome_name", "start_position",
                "end_position", "abspos"):
        assert col in df.columns, f"missing column {col}"

def test_hg20_bins_parquet_exists():
    df = pd.read_parquet(DATA / "hg20_220kb_bins.parquet")
    assert len(df) > 10_000
    assert {"chrom", "abspos"}.issubset(df.columns)

def test_cyclegenes_txt():
    lines = (DATA / "hg20_cycle_genes.txt").read_text().strip().splitlines()
    assert len(lines) > 100
```

- [ ] **Step 5: Run test → expect FAIL**

Run: `cd /media/jason/T7/rerbulid/pycopykat && uv run pytest tests/unit/test_reference_data.py -v`
Expected: FAIL (files don't exist yet).

- [ ] **Step 6: Run conversion script**

```bash
cd /media/jason/T7/rerbulid/pycopykat && Rscript scripts/convert_sysdata.R
```
Expected: 3 files created in `data/`.

- [ ] **Step 7: Run test → expect PASS**

```bash
uv run pytest tests/unit/test_reference_data.py -v
```
Expected: 3 passed.

- [ ] **Step 8: If column names differ from assumption, adjust test**

If the actual columns from `full.anno` are e.g. `chrom` vs `chromosome_name`, update both the test and document the canonical column names in `src/pycopykat/io/annotation.py` when building Task 3.1.

- [ ] **Step 9: Commit**

```bash
git add scripts/convert_sysdata.R data/*.parquet data/*.txt tests/unit/test_reference_data.py
git commit -m "data: convert hg20 sysdata.rda to parquet + cycle genes list"
```

---

## Task 1.3: Create `config.py` with typed CopykatConfig dataclass

**Files:**
- Create: `src/pycopykat/config.py`
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_config.py`:
```python
from pycopykat.config import CopykatConfig

def test_default_config():
    c = CopykatConfig()
    assert c.id_type == "Symbol"
    assert c.genome == "hg20"
    assert c.ngene_chr == 5
    assert c.min_gene_per_cell == 200
    assert c.low_dr == 0.05
    assert c.up_dr == 0.1
    assert c.win_size == 25
    assert c.ks_cut == 0.1
    assert c.distance == "euclidean"
    assert c.cell_line is False
    assert c.seed == 1234
    assert c.n_jobs == 1

def test_config_accepts_overrides():
    c = CopykatConfig(n_jobs=8, distance="pearson")
    assert c.n_jobs == 8
    assert c.distance == "pearson"

def test_config_rejects_invalid_distance():
    import pytest
    with pytest.raises(ValueError):
        CopykatConfig(distance="manhattan")  # not supported
```

- [ ] **Step 2: Run test → FAIL**

```bash
uv run pytest tests/unit/test_config.py -v
```

- [ ] **Step 3: Implement `config.py`**

Create `src/pycopykat/config.py`:
```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

IdType = Literal["Symbol", "Ensembl"]
Genome = Literal["hg20"]  # V1: mm10 deferred
Distance = Literal["euclidean", "pearson", "spearman"]
Backend = Literal["cpu"]  # V1 only

@dataclass(slots=True)
class CopykatConfig:
    id_type: IdType = "Symbol"
    genome: Genome = "hg20"
    cell_line: bool = False
    ngene_chr: int = 5
    min_gene_per_cell: int = 200
    low_dr: float = 0.05
    up_dr: float = 0.1
    win_size: int = 25
    ks_cut: float = 0.1
    distance: Distance = "euclidean"
    norm_cell_names: list[str] | None = None
    sam_name: str = ""
    output_dir: Path = field(default_factory=lambda: Path("."))
    output_seg: bool = False
    output_h5ad: bool = False
    plot_genes: bool = True
    n_jobs: int = 1
    seed: int = 1234
    backend: Backend = "cpu"

    def __post_init__(self) -> None:
        if self.distance not in ("euclidean", "pearson", "spearman"):
            raise ValueError(f"distance must be euclidean/pearson/spearman, got {self.distance}")
        if self.id_type not in ("Symbol", "Ensembl"):
            raise ValueError(f"id_type must be Symbol/Ensembl, got {self.id_type}")
        if self.genome != "hg20":
            raise ValueError(f"V1 only supports hg20; got {self.genome}")
        self.output_dir = Path(self.output_dir)
```

- [ ] **Step 4: Run test → PASS**

```bash
uv run pytest tests/unit/test_config.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/config.py tests/unit/test_config.py
git commit -m "feat: add CopykatConfig dataclass with validation"
```

---

## Task 1.4: Create `CopykatResult` dataclass

**Files:**
- Create: `src/pycopykat/result.py`
- Create: `tests/unit/test_result.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_result.py`:
```python
import numpy as np
import pandas as pd
from pycopykat.result import CopykatResult

def _toy_result():
    cna = pd.DataFrame(
        np.zeros((5, 3)),
        columns=["c1", "c2", "c3"],
        index=pd.MultiIndex.from_tuples(
            [(1, 100, 200), (1, 200, 300), (1, 300, 400), (2, 100, 200), (2, 200, 300)],
            names=["chrom", "start", "end"],
        ),
    )
    pred = pd.DataFrame(
        {"cell": ["c1", "c2", "c3"], "copykat.pred": ["aneuploid", "diploid", "diploid"]}
    )
    return CopykatResult(
        cna_mat=cna,
        prediction=pred,
        linkage=np.zeros((2, 4)),
        subclone=pd.Series(["c1"], index=["c1"]),
        warnings=("ok",),
    )

def test_result_has_expected_attrs():
    r = _toy_result()
    assert r.cna_mat.shape == (5, 3)
    assert r.prediction["copykat.pred"].tolist() == ["aneuploid", "diploid", "diploid"]
    assert r.warnings == ("ok",)
```

- [ ] **Step 2: Run test → FAIL**

```bash
uv run pytest tests/unit/test_result.py -v
```

- [ ] **Step 3: Implement `result.py`**

Create `src/pycopykat/result.py`:
```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass(slots=True)
class CopykatResult:
    """Output of pycopykat.copykat().

    cna_mat : bins x cells DataFrame with MultiIndex (chrom, start, end).
    prediction : cells x {cell, copykat.pred} DataFrame.
    linkage : scipy.cluster.hierarchy linkage matrix (n-1, 4).
    subclone : Series mapping aneuploid cell names → subclone label (int).
    warnings : tuple of human-readable warning strings.
    """
    cna_mat: pd.DataFrame
    prediction: pd.DataFrame
    linkage: np.ndarray
    subclone: pd.Series
    warnings: tuple[str, ...]

    def to_txt(self, output_dir: Path, sam_name: str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.cna_mat.reset_index().to_csv(
            output_dir / f"{sam_name}_copykat_CNA_results.txt",
            sep="\t", index=False,
        )
        self.prediction.to_csv(
            output_dir / f"{sam_name}_copykat_prediction.txt",
            sep="\t", index=False,
        )

    def to_h5ad(self, path: Path) -> None:
        import anndata as ad
        # cells x bins
        X = self.cna_mat.T.to_numpy()
        var = self.cna_mat.index.to_frame(index=False)
        obs = self.prediction.set_index("cell")
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.write_h5ad(path)
```

- [ ] **Step 4: Run test → PASS**

```bash
uv run pytest tests/unit/test_result.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/result.py tests/unit/test_result.py
git commit -m "feat: add CopykatResult dataclass with to_txt/to_h5ad"
```

---

# M2 — Numba Kernels

All kernels live in `src/pycopykat/kernels/`. Each kernel has:
1. A pure-Python reference implementation (for testing correctness)
2. A `@numba.njit(cache=True, parallel=True)` fast path
3. Unit tests comparing both on random inputs

## Task 2.1: `distances.py` — pairwise Euclidean / Pearson / Spearman

**Files:**
- Create: `src/pycopykat/kernels/distances.py`
- Create: `tests/unit/test_kernels_distances.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_kernels_distances.py`:
```python
import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from pycopykat.kernels.distances import pdist_euclidean, pdist_pearson, pdist_spearman

rng = np.random.default_rng(0)

def _rand(n=30, p=40):
    return rng.standard_normal((n, p))

def test_euclidean_matches_scipy():
    X = _rand()
    got = pdist_euclidean(X)
    want = pdist(X, "euclidean")
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)

def test_pearson_matches_scipy():
    X = _rand()
    got = pdist_pearson(X)  # 1 - r
    want = pdist(X, "correlation")  # scipy's "correlation" is 1 - Pearson
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)

def test_spearman_matches_scipy():
    X = _rand(n=20, p=30)
    got = pdist_spearman(X)
    n = X.shape[0]
    want = np.empty(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(X[i], X[j])
            want[k] = 1.0 - r
            k += 1
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)

def test_shapes():
    X = _rand(n=50, p=10)
    for fn in (pdist_euclidean, pdist_pearson, pdist_spearman):
        out = fn(X)
        assert out.shape == (50 * 49 // 2,)
```

- [ ] **Step 2: Run test → FAIL**

```bash
uv run pytest tests/unit/test_kernels_distances.py -v
```

- [ ] **Step 3: Implement `distances.py`**

Create `src/pycopykat/kernels/distances.py`:
```python
"""Pairwise distance kernels. All take (n, p) arrays and return condensed
distance vectors of length n*(n-1)/2 (scipy convention)."""
from __future__ import annotations
import numpy as np
from numba import njit, prange
from scipy.spatial.distance import pdist

def pdist_euclidean(X: np.ndarray) -> np.ndarray:
    """Thin wrapper over scipy (already C-optimized)."""
    return pdist(np.ascontiguousarray(X, dtype=np.float64), metric="euclidean")

@njit(cache=True, parallel=True, fastmath=True)
def _pdist_pearson_kernel(X: np.ndarray) -> np.ndarray:
    n, p = X.shape
    # Center rows
    means = np.empty(n)
    stds = np.empty(n)
    for i in prange(n):
        m = 0.0
        for k in range(p):
            m += X[i, k]
        m /= p
        means[i] = m
        s = 0.0
        for k in range(p):
            d = X[i, k] - m
            s += d * d
        stds[i] = np.sqrt(s / p)
    out = np.empty(n * (n - 1) // 2)
    # Fill (i,j) pairs. Compute per-row index offsets.
    # cumulative pairs before row i: i*(2n - i - 1) / 2
    for i in prange(n - 1):
        offset = i * (2 * n - i - 1) // 2
        for j in range(i + 1, n):
            mi, mj = means[i], means[j]
            si, sj = stds[i], stds[j]
            if si == 0.0 or sj == 0.0:
                out[offset + (j - i - 1)] = 1.0
                continue
            dot = 0.0
            for k in range(p):
                dot += (X[i, k] - mi) * (X[j, k] - mj)
            r = dot / (p * si * sj)
            out[offset + (j - i - 1)] = 1.0 - r
    return out

def pdist_pearson(X: np.ndarray) -> np.ndarray:
    return _pdist_pearson_kernel(np.ascontiguousarray(X, dtype=np.float64))

def pdist_spearman(X: np.ndarray) -> np.ndarray:
    """Rank-transform rows, then delegate to Pearson."""
    from scipy.stats import rankdata
    R = np.apply_along_axis(rankdata, 1, X)
    return pdist_pearson(R)
```

- [ ] **Step 4: Run test → PASS**

```bash
uv run pytest tests/unit/test_kernels_distances.py -v
```

If the Numba-compiled Pearson output disagrees with scipy: check the index formula; for n rows, row i's pair-block starts at `i*(2n-i-1)//2` and has length `n-1-i`.

- [ ] **Step 5: Add benchmark (optional now, required later)**

Append to the test file:
```python
@pytest.mark.benchmark(group="pearson")
def test_bench_pearson_1k(benchmark):
    X = np.random.default_rng(0).standard_normal((1000, 500))
    benchmark(pdist_pearson, X)
```

- [ ] **Step 6: Commit**

```bash
git add src/pycopykat/kernels/distances.py tests/unit/test_kernels_distances.py
git commit -m "feat(kernels): pdist euclidean/pearson/spearman with numba"
```

---

## Task 2.2: `kalman.py` — per-cell RTS smoother

R uses `dlm::dlmModPoly(order=1, dV=0.16, dW=0.001)` + `dlmSmooth`. For order-1 polynomial, the state is scalar; filter/smoother can be written in closed form.

**Files:**
- Create: `src/pycopykat/kernels/kalman.py`
- Create: `tests/unit/test_kernels_kalman.py`

Model equations (order-1 local level):
```
x_t = x_{t-1} + w_t,   w_t ~ N(0, dW)
y_t = x_t + v_t,       v_t ~ N(0, dV)
```
RTS: forward filter + backward smooth. Closed-form scalar updates.

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_kernels_kalman.py`:
```python
import subprocess
import numpy as np
import pytest
from pycopykat.kernels.kalman import kalman_smooth, kalman_smooth_matrix

rng = np.random.default_rng(0)

def _simulate(n=200, dW=0.001, dV=0.16, seed=0):
    r = np.random.default_rng(seed)
    x = np.cumsum(r.normal(0, np.sqrt(dW), size=n))
    y = x + r.normal(0, np.sqrt(dV), size=n)
    return y

def test_smoother_shape():
    y = _simulate()
    s = kalman_smooth(y, dV=0.16, dW=0.001)
    assert s.shape == y.shape

def test_smoother_reduces_variance():
    y = _simulate(n=500)
    s = kalman_smooth(y, dV=0.16, dW=0.001)
    # Smoothed series should have much smaller first-difference variance
    assert np.var(np.diff(s)) < np.var(np.diff(y)) * 0.5

def test_matches_r_dlm_smooth(tmp_path):
    """Bit-close match to dlm::dlmSmooth on same input."""
    y = _simulate(n=50, seed=42)
    rscript = tmp_path / "run.R"
    ypath = tmp_path / "y.txt"
    opath = tmp_path / "out.txt"
    np.savetxt(ypath, y)
    rscript.write_text(f"""
      library(dlm)
      y <- scan("{ypath}", quiet=TRUE)
      m <- dlmModPoly(order=1, dV=0.16, dW=0.001)
      s <- dlmSmooth(y, m)$s
      s <- s[2:length(s)]
      s <- s - mean(s)
      write.table(s, "{opath}", row.names=FALSE, col.names=FALSE)
    """)
    subprocess.run(["Rscript", str(rscript)], check=True, capture_output=True)
    r_s = np.loadtxt(opath)
    py_s = kalman_smooth(y, dV=0.16, dW=0.001)
    py_s = py_s - py_s.mean()  # R copykat re-centers after smoothing
    np.testing.assert_allclose(py_s, r_s, rtol=1e-4, atol=1e-4)

def test_matrix_version_equivalent_to_loop():
    Y = rng.standard_normal((100, 20))
    loop = np.column_stack([kalman_smooth(Y[:, j], 0.16, 0.001) for j in range(20)])
    mat = kalman_smooth_matrix(Y, 0.16, 0.001)
    np.testing.assert_allclose(mat, loop, rtol=1e-10, atol=1e-12)
```

- [ ] **Step 2: Run test → FAIL**

```bash
uv run pytest tests/unit/test_kernels_kalman.py -v
```

- [ ] **Step 3: Implement `kalman.py`**

Create `src/pycopykat/kernels/kalman.py`:
```python
"""Order-1 polynomial (local level) Kalman RTS smoother.

Matches dlm::dlmModPoly(order=1, dV, dW) + dlmSmooth.
Scalar state; closed-form per-step recursions.
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True)
def _rts_smooth_1d(y: np.ndarray, dV: float, dW: float) -> np.ndarray:
    """
    Forward filter + backward RTS smoother on scalar state with
      x_t = x_{t-1} + w_t, w ~ N(0, dW)
      y_t = x_t + v_t,     v ~ N(0, dV)
    Initial: x_0 ~ N(0, 1e7) (diffuse).
    """
    n = y.shape[0]
    # Forward filter
    m_fwd = np.empty(n + 1)
    C_fwd = np.empty(n + 1)
    a_fwd = np.empty(n + 1)
    R_fwd = np.empty(n + 1)
    m_fwd[0] = 0.0
    C_fwd[0] = 1e7
    for t in range(1, n + 1):
        a = m_fwd[t - 1]
        R = C_fwd[t - 1] + dW
        f = a                      # predicted y
        Q = R + dV
        K = R / Q
        e = y[t - 1] - f
        m_fwd[t] = a + K * e
        C_fwd[t] = R - K * R       # = R * dV / Q
        a_fwd[t] = a
        R_fwd[t] = R
    # Backward smoother (RTS)
    s = np.empty(n + 1)
    s[n] = m_fwd[n]
    for t in range(n - 1, -1, -1):
        if t == 0:
            J = C_fwd[0] / (C_fwd[0] + dW)
            s[0] = m_fwd[0] + J * (s[1] - (m_fwd[0]))
        else:
            J = C_fwd[t] / R_fwd[t + 1]
            s[t] = m_fwd[t] + J * (s[t + 1] - a_fwd[t + 1])
    return s

def kalman_smooth(y: np.ndarray, dV: float = 0.16, dW: float = 0.001) -> np.ndarray:
    """Smooth a 1D series. Returns smoothed series of same length as y
    (R copykat drops s[1] so s[2:] is length n; we replicate that)."""
    s_full = _rts_smooth_1d(np.ascontiguousarray(y, dtype=np.float64), dV, dW)
    return s_full[1:]  # drop the prior x_0 estimate

@njit(cache=True, parallel=True, fastmath=True)
def _smooth_matrix_kernel(Y: np.ndarray, dV: float, dW: float) -> np.ndarray:
    g, c = Y.shape
    out = np.empty_like(Y)
    for j in prange(c):
        s_full = _rts_smooth_1d(Y[:, j].copy(), dV, dW)
        for i in range(g):
            out[i, j] = s_full[i + 1]
    return out

def kalman_smooth_matrix(Y: np.ndarray, dV: float = 0.16, dW: float = 0.001) -> np.ndarray:
    """Apply smoother column-wise (each cell independent). Parallel over cells."""
    return _smooth_matrix_kernel(np.ascontiguousarray(Y, dtype=np.float64), dV, dW)
```

- [ ] **Step 4: Run test → PASS**

```bash
uv run pytest tests/unit/test_kernels_kalman.py -v
```

If `test_matches_r_dlm_smooth` fails with large diff, investigate initial prior variance (R dlm's default for `dlmModPoly` uses diffuse prior `C0 = 1e7 * I` — verify by reading `?dlmModPoly`).

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/kernels/kalman.py tests/unit/test_kernels_kalman.py
git commit -m "feat(kernels): numba Kalman RTS smoother matching dlm::dlmSmooth"
```

---

## Task 2.3: `mcmc_pg.py` — Poisson-Gamma conjugate sampler

MCMCpack::MCpoissongamma verified: `rgamma(mc, alpha + sum(y), beta + n)`.
R's rgamma(shape, rate) → NumPy gamma(shape, scale=1/rate).

**Files:**
- Create: `src/pycopykat/kernels/mcmc_pg.py`
- Create: `tests/unit/test_kernels_mcmc_pg.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_kernels_mcmc_pg.py`:
```python
import subprocess
import numpy as np
import pytest
from scipy.stats import ks_2samp
from pycopykat.kernels.mcmc_pg import pg_posterior_mean, pg_posterior_samples

rng = np.random.default_rng(42)

def test_posterior_samples_shape():
    y = rng.poisson(5.0, size=30).astype(np.float64)
    samp = pg_posterior_samples(y, alpha=1.0, beta=1.0, mc=1000, seed=0)
    assert samp.shape == (1000,)
    # Analytic posterior mean = (a + sum(y)) / (b + n)
    expected = (1.0 + y.sum()) / (1.0 + y.size)
    assert abs(samp.mean() - expected) < 0.5

def test_matches_mcmcpack(tmp_path):
    y = rng.poisson(5.0, size=40).astype(np.float64)
    np.savetxt(tmp_path / "y.txt", y)
    rscript = tmp_path / "run.R"
    rscript.write_text(f"""
      suppressMessages(library(MCMCpack))
      y <- scan("{tmp_path}/y.txt", quiet=TRUE)
      set.seed(123)
      s <- as.numeric(MCpoissongamma(y, 1.0, 1.0, mc=2000))
      write.table(s, "{tmp_path}/s.txt", row.names=FALSE, col.names=FALSE)
    """)
    subprocess.run(["Rscript", str(rscript)], check=True, capture_output=True)
    r_samp = np.loadtxt(tmp_path / "s.txt")
    py_samp = pg_posterior_samples(y, alpha=1.0, beta=1.0, mc=2000, seed=123)
    # Two i.i.d. Gamma samples of same size should have similar distributions
    stat, p = ks_2samp(r_samp, py_samp)
    assert p > 0.01, f"KS p={p:.3f} too small (stat={stat:.3f})"

def test_mean_batched():
    Y_segments = [rng.poisson(l, size=20).astype(np.float64) for l in (1, 5, 20)]
    means = np.array([pg_posterior_mean(y, 1.0, 1.0, mc=1000, seed=i)
                      for i, y in enumerate(Y_segments)])
    analytic = np.array([(1.0 + y.sum()) / (1.0 + y.size) for y in Y_segments])
    np.testing.assert_allclose(means, analytic, rtol=0.1)
```

- [ ] **Step 2: Run test → FAIL**

```bash
uv run pytest tests/unit/test_kernels_mcmc_pg.py -v
```

- [ ] **Step 3: Implement `mcmc_pg.py`**

Create `src/pycopykat/kernels/mcmc_pg.py`:
```python
"""Poisson-Gamma conjugate posterior sampling.

MCMCpack::MCpoissongamma(y, alpha, beta, mc) = rgamma(mc, alpha+sum(y), beta+n).
R's rgamma(shape, rate) equals numpy.random.Generator.gamma(shape, scale=1/rate).
"""
from __future__ import annotations
import numpy as np
from numba import njit

def pg_posterior_samples(
    y: np.ndarray, alpha: float, beta: float, mc: int, seed: int
) -> np.ndarray:
    """Draw `mc` i.i.d. samples from Gamma(alpha + sum(y), rate=beta + n)."""
    rng = np.random.default_rng(seed)
    shape = alpha + float(y.sum())
    scale = 1.0 / (beta + y.size)
    return rng.gamma(shape=shape, scale=scale, size=mc)

def pg_posterior_mean(
    y: np.ndarray, alpha: float, beta: float, mc: int, seed: int
) -> float:
    """Monte-Carlo estimate of posterior mean lambda."""
    return float(pg_posterior_samples(y, alpha, beta, mc, seed).mean())

# Numba JIT path — callers pre-generate uniforms (Numba has limited rng support)
@njit(cache=True, fastmath=True)
def pg_posterior_mean_from_shape_scale(shape: float, scale: float, draws: np.ndarray) -> float:
    """Given precomputed uniform(0,1) draws, return the Monte-Carlo posterior mean
    via inverse-CDF Gamma sampling using Marsaglia-Tsang shape>=1 trick.
    For shape<1, we fall back to boosting: sample at shape+1 and multiply by U^(1/shape).
    """
    # Marsaglia-Tsang: sample Gamma(shape, 1), multiply by scale.
    n = draws.shape[0]
    total = 0.0
    # We'll use Python layer for simplicity in V1; Numba accelerates only aggregation.
    # Fall back: caller must pre-sample via numpy. Keep this stub for API symmetry.
    for i in range(n):
        total += draws[i] * scale
    return total / n
```

Note: V1 uses the NumPy-side Gamma sampler; the Numba JIT path is reserved for future micro-optimization but is currently a straight aggregator. The speed-critical outer loop (over breakpoint candidates × clusters × cells) will be parallelized at the caller level with `joblib` in Task 5.2.

- [ ] **Step 4: Run test → PASS**

```bash
uv run pytest tests/unit/test_kernels_mcmc_pg.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/kernels/mcmc_pg.py tests/unit/test_kernels_mcmc_pg.py
git commit -m "feat(kernels): Poisson-Gamma conjugate posterior sampler"
```

---

## Task 2.4: `adjust.py` — per-cell threshold adjustment kernel

R copykat.R lines 378–391: for each bin value, if `|v - base[bin]| < 0.25 * cf.h[bin]`, replace with `mean(cell)`. Then re-center.

**Files:**
- Create: `src/pycopykat/kernels/adjust.py`
- Create: `tests/unit/test_kernels_adjust.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
from pycopykat.kernels.adjust import adjust_threshold

def test_adjust_mutes_near_baseline():
    # g=5 bins, c=2 cells
    X = np.array([[0.0, 0.0, 1.0, 2.0, 3.0],
                  [0.1, 0.1, 1.1, 2.1, 3.1]]).T  # (5, 2)
    base = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    sd = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    out = adjust_threshold(X, base, sd, factor=0.25)
    # |X - base| < 0.25 → replace by col mean
    col_means = X.mean(axis=0)
    for j in range(2):
        for i in range(5):
            if abs(X[i, j] - base[i]) < 0.25 * sd[i]:
                assert out[i, j] == pytest.approx(col_means[j])
            else:
                assert out[i, j] == pytest.approx(X[i, j])
```

Add `import pytest` at top.

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

Create `src/pycopykat/kernels/adjust.py`:
```python
from __future__ import annotations
import numpy as np
from numba import njit, prange

@njit(cache=True, parallel=True, fastmath=True)
def _adjust_kernel(X: np.ndarray, base: np.ndarray, sd: np.ndarray, factor: float) -> np.ndarray:
    g, c = X.shape
    out = np.empty_like(X)
    for j in prange(c):
        col_mean = 0.0
        for i in range(g):
            col_mean += X[i, j]
        col_mean /= g
        thr = factor
        for i in range(g):
            if abs(X[i, j] - base[i]) < thr * sd[i]:
                out[i, j] = col_mean
            else:
                out[i, j] = X[i, j]
    return out

def adjust_threshold(X: np.ndarray, base: np.ndarray, sd: np.ndarray, factor: float = 0.25) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float64)
    base = np.ascontiguousarray(base, dtype=np.float64)
    sd = np.ascontiguousarray(sd, dtype=np.float64)
    return _adjust_kernel(X, base, sd, float(factor))
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/kernels/adjust.py tests/unit/test_kernels_adjust.py
git commit -m "feat(kernels): per-cell threshold adjustment"
```

---

## Task 2.5: Kernels package `__init__.py`

- [ ] **Step 1: Overwrite `src/pycopykat/kernels/__init__.py`**

```python
from pycopykat.kernels.distances import pdist_euclidean, pdist_pearson, pdist_spearman
from pycopykat.kernels.kalman import kalman_smooth, kalman_smooth_matrix
from pycopykat.kernels.mcmc_pg import pg_posterior_samples, pg_posterior_mean
from pycopykat.kernels.adjust import adjust_threshold

__all__ = [
    "pdist_euclidean", "pdist_pearson", "pdist_spearman",
    "kalman_smooth", "kalman_smooth_matrix",
    "pg_posterior_samples", "pg_posterior_mean",
    "adjust_threshold",
]
```

- [ ] **Step 2: Commit**

```bash
git add src/pycopykat/kernels/__init__.py
git commit -m "chore: export kernels public API"
```

---

# M3 — Pipeline Front End

## Task 3.1: `io/annotation.py` — hg20 gene annotation

**Files:**
- Create: `src/pycopykat/io/annotation.py`
- Create: `tests/unit/test_io_annotation.py`

- [ ] **Step 1: Write failing test**

```python
import pandas as pd
from pycopykat.io.annotation import load_hg20_annotation, annotate_genes

def test_load_returns_dataframe_with_expected_cols():
    ann = load_hg20_annotation()
    for col in ("hgnc_symbol", "ensembl_gene_id", "chromosome_name",
                "start_position", "end_position", "abspos"):
        assert col in ann.columns

def test_annotate_genes_symbol():
    # Build a toy expression matrix whose index matches hg20 symbols
    ann = load_hg20_annotation()
    # pick 5 HGNC symbols known to exist
    syms = ann["hgnc_symbol"].dropna().unique()[:5].tolist()
    import numpy as np
    mat = pd.DataFrame(np.zeros((5, 3)), index=syms, columns=["c1", "c2", "c3"])
    out = annotate_genes(mat, id_type="Symbol", genome="hg20")
    assert "chromosome_name" in out.columns
    assert out.shape[0] <= 5
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

Create `src/pycopykat/io/annotation.py`:
```python
from __future__ import annotations
from pathlib import Path
from importlib.resources import files
import pandas as pd

def _data_dir() -> Path:
    # data/ sits alongside the installed package root
    return Path(__file__).resolve().parents[3] / "data"

def load_hg20_annotation() -> pd.DataFrame:
    path = _data_dir() / "hg20_gene_anno.parquet"
    return pd.read_parquet(path)

def load_hg20_cycle_genes() -> list[str]:
    path = _data_dir() / "hg20_cycle_genes.txt"
    return [s.strip() for s in path.read_text().splitlines() if s.strip()]

def load_hg20_bins() -> pd.DataFrame:
    path = _data_dir() / "hg20_220kb_bins.parquet"
    return pd.read_parquet(path)

def annotate_genes(
    expr: pd.DataFrame, *, id_type: str = "Symbol", genome: str = "hg20",
) -> pd.DataFrame:
    """
    expr : genes × cells with gene identifiers in the index.
    Returns a DataFrame with 7 annotation columns + original cell columns,
    sorted by abspos.
    """
    if genome != "hg20":
        raise NotImplementedError("V1 only supports hg20")
    ann = load_hg20_annotation()
    key = "hgnc_symbol" if id_type == "Symbol" else "ensembl_gene_id"
    ann = ann.dropna(subset=[key]).drop_duplicates(subset=[key])
    merged = ann.merge(
        expr.rename_axis(index=key).reset_index(),
        on=key, how="inner",
    )
    # Remove HLA-* and cell cycle genes
    cyc = set(load_hg20_cycle_genes())
    is_hla = merged["hgnc_symbol"].astype(str).str.startswith("HLA-")
    is_cyc = merged["hgnc_symbol"].isin(cyc)
    merged = merged.loc[~(is_hla | is_cyc)].copy()
    merged = merged.sort_values("abspos", kind="mergesort").reset_index(drop=True)
    return merged
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/io/annotation.py tests/unit/test_io_annotation.py
git commit -m "feat(io): hg20 gene annotation loader and annotate_genes"
```

---

## Task 3.2: `preprocess/filter.py` — stage-1 cell/gene filtering

**Files:**
- Create: `src/pycopykat/preprocess/filter.py`
- Create: `tests/unit/test_preprocess_filter.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pandas as pd
from pycopykat.preprocess.filter import filter_cells_and_genes

def test_filters_low_gene_cells():
    # 10 genes × 4 cells; cell 0 has only 50 genes expressed → below default 200
    rng = np.random.default_rng(0)
    mat = pd.DataFrame(rng.poisson(5, size=(100, 4)),
                       index=[f"g{i}" for i in range(100)],
                       columns=["c1", "c2", "c3", "c4"])
    mat["c1"] = 0
    mat.iloc[:50, mat.columns.get_loc("c1")] = 1  # 50 non-zero genes
    out, stats = filter_cells_and_genes(mat, min_gene_per_cell=60, low_dr=0.05)
    assert "c1" not in out.columns
    assert stats["n_cells_dropped"] == 1
    assert stats["data_quality"] in ("ok", "low")

def test_filters_low_dr_genes():
    rng = np.random.default_rng(1)
    mat = pd.DataFrame(rng.poisson(1, size=(50, 1000)),
                       index=[f"g{i}" for i in range(50)],
                       columns=[f"c{i}" for i in range(1000)])
    mat.iloc[0, :] = 0  # gene 0 always zero → DR=0 → dropped at low_dr=0.05
    out, _ = filter_cells_and_genes(mat, min_gene_per_cell=1, low_dr=0.05)
    assert "g0" not in out.index
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

```python
# src/pycopykat/preprocess/filter.py
from __future__ import annotations
import numpy as np
import pandas as pd

def filter_cells_and_genes(
    mat: pd.DataFrame, *, min_gene_per_cell: int, low_dr: float,
) -> tuple[pd.DataFrame, dict]:
    """
    mat : genes × cells raw count DataFrame.
    Returns (filtered_mat, stats_dict).
    Matches R copykat.R stage 1 (lines 41–60).
    """
    X = mat.to_numpy()
    genes_per_cell = (X > 0).sum(axis=0)
    keep_cells = genes_per_cell > min_gene_per_cell
    if keep_cells.sum() == 0:
        raise ValueError("no cells have more than min_gene_per_cell genes")
    mat2 = mat.loc[:, keep_cells]
    n_cells_dropped = int((~keep_cells).sum())

    X2 = mat2.to_numpy()
    der = (X2 > 0).sum(axis=1) / X2.shape[1]
    keep_genes = der > low_dr
    mat3 = mat2.loc[keep_genes]
    quality = "ok" if mat3.shape[0] >= 7000 else "low"
    return mat3, {
        "n_cells_dropped": n_cells_dropped,
        "n_genes_kept": int(keep_genes.sum()),
        "data_quality": quality,
    }
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/preprocess/filter.py tests/unit/test_preprocess_filter.py
git commit -m "feat(preprocess): cell/gene QC filtering matching R copykat stage 1"
```

---

## Task 3.3: `preprocess/filter.py` — chromosome-coverage secondary filter

R copykat.R lines 84–103: per cell, require (#chromosomes present - ngene_chr genes) coverage. Add this as a second function in the same module.

**Files:**
- Modify: `src/pycopykat/preprocess/filter.py`
- Modify: `tests/unit/test_preprocess_filter.py`

- [ ] **Step 1: Add failing test**

Append to `test_preprocess_filter.py`:
```python
from pycopykat.preprocess.filter import filter_cells_by_chrom_coverage

def test_chrom_coverage_requires_min_per_chrom():
    # 3 chroms × 10 genes each, 2 cells
    idx = pd.MultiIndex.from_arrays(
        [[1]*10 + [2]*10 + [3]*10, list(range(30))],
        names=["chrom", "gene"],
    )
    mat = pd.DataFrame(np.ones((30, 2)), index=idx, columns=["c1", "c2"])
    # c2 has no expression on chrom 3
    mat.loc[(3, slice(None)), "c2"] = 0
    chrom = mat.index.get_level_values("chrom").to_numpy()
    out, dropped = filter_cells_by_chrom_coverage(mat.to_numpy(), chrom, ngene_chr=5)
    assert dropped == [1]   # index of c2
    assert out.shape[1] == 1
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

Append to `filter.py`:
```python
def filter_cells_by_chrom_coverage(
    X: np.ndarray, chrom: np.ndarray, *, ngene_chr: int,
) -> tuple[np.ndarray, list[int]]:
    """
    X      : genes × cells annotated expression (ordered by abspos).
    chrom  : length-genes array of chromosome ids (int).
    Returns (filtered_X, dropped_cell_indices).
    A cell is dropped if for any chromosome it has fewer than ngene_chr
    non-zero genes, OR if the total non-zero genes < 5.
    """
    g, c = X.shape
    unique_chroms = np.unique(chrom)
    drop = []
    for j in range(c):
        nz = X[:, j] > 0
        if nz.sum() < 5:
            drop.append(j)
            continue
        ok = True
        for k in unique_chroms:
            if (nz & (chrom == k)).sum() < ngene_chr:
                ok = False
                break
        if not ok:
            drop.append(j)
    keep = [j for j in range(c) if j not in set(drop)]
    return X[:, keep], drop
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/preprocess/filter.py tests/unit/test_preprocess_filter.py
git commit -m "feat(preprocess): chromosome-coverage secondary filter"
```

---

## Task 3.4: `preprocess/normalize.py` — VST + centering

R copykat.R lines 105–107: `norm.mat = log(sqrt(x) + sqrt(x+1))`, then center each column (cell) by subtracting its mean.

**Files:**
- Create: `src/pycopykat/preprocess/normalize.py`
- Create: `tests/unit/test_preprocess_normalize.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
from pycopykat.preprocess.normalize import vst_center

def test_vst_formula_on_small_case():
    x = np.array([[0.0, 1.0], [4.0, 9.0]])
    want_raw = np.log(np.sqrt(x) + np.sqrt(x + 1))
    want = want_raw - want_raw.mean(axis=0)
    got = vst_center(x)
    np.testing.assert_allclose(got, want, rtol=1e-10)

def test_columns_are_mean_zero():
    rng = np.random.default_rng(0)
    x = rng.poisson(5, size=(100, 10)).astype(float)
    out = vst_center(x)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

```python
# src/pycopykat/preprocess/normalize.py
from __future__ import annotations
import numpy as np

def vst_center(x: np.ndarray) -> np.ndarray:
    """Variance-stabilizing transform + per-column (cell) centering.
    Matches R copykat: norm = log(sqrt(x) + sqrt(x+1)); then x -= mean(x) per col.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.log(np.sqrt(x) + np.sqrt(x + 1.0))
    y -= y.mean(axis=0, keepdims=True)
    return y
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/preprocess/normalize.py tests/unit/test_preprocess_normalize.py
git commit -m "feat(preprocess): VST + per-cell centering"
```

---

## Task 3.5: `preprocess/smooth.py` — wrap Kalman kernel with post-center

R: after smoothing, `x <- x[2:length(x)]; x <- x - mean(x)`.

**Files:**
- Create: `src/pycopykat/preprocess/smooth.py`
- Create: `tests/unit/test_preprocess_smooth.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
from pycopykat.preprocess.smooth import smooth_cells

def test_output_shape_and_centered():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 8))
    S = smooth_cells(X)
    assert S.shape == X.shape
    np.testing.assert_allclose(S.mean(axis=0), 0.0, atol=1e-10)
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

```python
# src/pycopykat/preprocess/smooth.py
from __future__ import annotations
import numpy as np
from pycopykat.kernels.kalman import kalman_smooth_matrix

def smooth_cells(X: np.ndarray, dV: float = 0.16, dW: float = 0.001) -> np.ndarray:
    """Per-cell Kalman smoothing then re-center (matches R copykat step 3)."""
    S = kalman_smooth_matrix(X, dV=dV, dW=dW)
    S -= S.mean(axis=0, keepdims=True)
    return S
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/preprocess/smooth.py tests/unit/test_preprocess_smooth.py
git commit -m "feat(preprocess): per-cell Kalman smoothing wrapper"
```

---

# M4 — Baseline Estimation

Three baseline modes all share a common hierarchical clustering step. Build the shared helper first.

## Task 4.1: `baseline/_shared.py` — cluster-reduce helper

R iterates: starting k=6, cutree, if any cluster has <min.cells, decrement k and retry; then EM on each cluster.

**Files:**
- Create: `src/pycopykat/baseline/_shared.py`
- Create: `tests/unit/test_baseline_shared.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
from pycopykat.baseline._shared import ward_cluster_with_min_size

def test_returns_labels_honoring_min_size():
    rng = np.random.default_rng(0)
    # 4 well-separated blobs of 25 cells each
    X = np.vstack([
        rng.normal(loc=0,  scale=0.1, size=(25, 10)),
        rng.normal(loc=5,  scale=0.1, size=(25, 10)),
        rng.normal(loc=10, scale=0.1, size=(25, 10)),
        rng.normal(loc=15, scale=0.1, size=(25, 10)),
    ])
    labels, Z = ward_cluster_with_min_size(X, initial_k=6, min_cells=10, distance="euclidean")
    assert Z.shape[0] == 99  # n-1 linkage rows
    for lab in np.unique(labels):
        assert (labels == lab).sum() >= 10
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

```python
# src/pycopykat/baseline/_shared.py
from __future__ import annotations
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from pycopykat.kernels.distances import pdist_euclidean, pdist_pearson, pdist_spearman

_DIST_FN = {
    "euclidean": pdist_euclidean,
    "pearson": pdist_pearson,
    "spearman": pdist_spearman,
}

def ward_cluster_with_min_size(
    X: np.ndarray, *, initial_k: int = 6, min_cells: int = 10,
    distance: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    X : (n, p) — n points to cluster.
    Returns (labels, linkage_matrix). Starts k=initial_k and decrements until
    all clusters have ≥ min_cells; k >= 1 always.
    """
    d = _DIST_FN[distance](X)
    Z = linkage(d, method="ward")
    k = initial_k
    while k > 1:
        labels = fcluster(Z, t=k, criterion="maxclust")
        counts = np.bincount(labels)[1:]
        if counts.min() >= min_cells:
            return labels, Z
        k -= 1
    return np.ones(X.shape[0], dtype=int), Z
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/baseline/_shared.py tests/unit/test_baseline_shared.py
git commit -m "feat(baseline): shared Ward-cluster-with-min-size helper"
```

---

## Task 4.2: `baseline/auto.py` — mixture-EM-based auto baseline

Mirrors R `baseline.norm.cl`. Uses sklearn GaussianMixture; initial means=[-0.2, 0, 0.2].

**Files:**
- Create: `src/pycopykat/baseline/auto.py`
- Create: `tests/unit/test_baseline_auto.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
from pycopykat.baseline.auto import baseline_norm_cl

def test_identifies_low_variance_cluster_as_baseline():
    rng = np.random.default_rng(0)
    g = 500
    # 30 diploid-like cells (low noise) + 30 aneuploid-like (high noise + shift)
    dip = rng.normal(0, 0.05, size=(g, 30))
    ane = rng.normal(0.3, 0.3, size=(g, 30))
    X = np.hstack([dip, ane])
    cell_names = [f"d{i}" for i in range(30)] + [f"a{i}" for i in range(30)]
    result = baseline_norm_cl(X, cell_names=cell_names, min_cells=5, n_jobs=1, seed=1234)
    # All diploid names should be in the result's preN
    assert set(cell_names[:30]).issubset(set(result.preN))
    assert result.warning in ("ok", "unclassified.prediction")
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

```python
# src/pycopykat/baseline/auto.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples
from scipy.stats import f as f_dist
from pycopykat.baseline._shared import ward_cluster_with_min_size
from pycopykat.kernels.distances import pdist_euclidean

@dataclass(slots=True)
class BaselineResult:
    basel: np.ndarray
    preN: list[str]
    warning: str
    labels: np.ndarray

def baseline_norm_cl(
    X: np.ndarray, cell_names: list[str], *,
    min_cells: int = 5, n_jobs: int = 1, seed: int = 1234,
) -> BaselineResult:
    """
    X : (genes, cells) smoothed matrix.
    Returns baseline vector (per-gene median of inferred diploid cluster).
    Matches R copykat baseline.norm.cl.
    """
    labels, _Z = ward_cluster_with_min_size(
        X.T, initial_k=6, min_cells=min_cells, distance="euclidean"
    )

    # Per-cluster EM; record sigma1 (lowest-variance component)
    sigmas = []
    for lab in np.unique(labels):
        data = X[:, labels == lab].ravel()
        sx = max(0.05, 0.5 * data.std(ddof=1))
        gmm = GaussianMixture(
            n_components=3,
            means_init=np.array([[-0.2], [0.0], [0.2]]),
            precisions_init=np.array([[1.0 / sx**2]] * 3).reshape(3, 1, 1),
            covariance_type="full",
            max_iter=1000,
            random_state=seed,
        )
        gmm.fit(data.reshape(-1, 1))
        stds = np.sqrt(gmm.covariances_.ravel())
        sigmas.append(stds.min())

    sigmas = np.array(sigmas)

    # Silhouette + F-test confidence check
    d = pdist_euclidean(X.T)
    from scipy.spatial.distance import squareform
    D = squareform(d)
    sil = silhouette_samples(D, labels, metric="precomputed").mean()
    SDM = sigmas
    n = X.shape[0]
    pdt = 1.0 - f_dist.cdf(SDM.max() ** 2 / SDM.min() ** 2, n, n)

    warning = "ok"
    if sil <= 0.15 or pdt > 0.05:
        warning = "unclassified.prediction"

    diploid_lab = np.unique(labels)[int(np.argmin(sigmas))]
    diploid_mask = labels == diploid_lab
    basel = np.median(X[:, diploid_mask], axis=1)
    preN = [cn for cn, m in zip(cell_names, diploid_mask) if m]
    return BaselineResult(basel=basel, preN=preN, warning=warning, labels=labels)
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/baseline/auto.py tests/unit/test_baseline_auto.py
git commit -m "feat(baseline): auto mode via cluster-wise GMM (baseline.norm.cl)"
```

---

## Task 4.3: `baseline/gmm.py` — per-cell GMM fallback

Mirrors R `baseline.GMM`.

**Files:**
- Create: `src/pycopykat/baseline/gmm.py`
- Create: `tests/unit/test_baseline_gmm.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
from pycopykat.baseline.gmm import baseline_gmm

def test_finds_diploid_cells_in_mixture():
    rng = np.random.default_rng(0)
    g = 500
    dip = rng.normal(0, 0.05, size=(g, 20))
    ane = rng.normal(0.5, 0.5, size=(g, 20))
    X = np.hstack([dip, ane])
    names = [f"d{i}" for i in range(20)] + [f"a{i}" for i in range(20)]
    res = baseline_gmm(X, cell_names=names, max_normal=5, mu_cut=0.05,
                      nfraq_cut=0.99, seed=1234)
    assert len(res.preN) >= 3
    # Diploid names should dominate
    assert sum(n.startswith("d") for n in res.preN) >= len(res.preN) * 0.6
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

```python
# src/pycopykat/baseline/gmm.py
from __future__ import annotations
import numpy as np
from sklearn.mixture import GaussianMixture
from pycopykat.baseline.auto import BaselineResult

def _is_diploid(col: np.ndarray, mu_cut: float, nfraq_cut: float, seed: int) -> bool:
    sx = max(0.05, 0.5 * col.std(ddof=1))
    gmm = GaussianMixture(
        n_components=3,
        means_init=np.array([[-0.2], [0.0], [0.2]]),
        precisions_init=np.array([[1.0 / sx**2]] * 3).reshape(3, 1, 1),
        covariance_type="full", max_iter=1000, random_state=seed,
    )
    gmm.fit(col.reshape(-1, 1))
    mus = gmm.means_.ravel()
    weights = gmm.weights_.ravel()
    frq = float(weights[np.abs(mus) <= mu_cut].sum())
    return frq > nfraq_cut

def baseline_gmm(
    X: np.ndarray, cell_names: list[str], *,
    max_normal: int = 5, mu_cut: float = 0.05, nfraq_cut: float = 0.99,
    seed: int = 1234,
) -> BaselineResult:
    """Per-cell GMM fallback (R baseline.GMM)."""
    g, c = X.shape
    diploid_idx: list[int] = []
    for j in range(c):
        if _is_diploid(X[:, j], mu_cut, nfraq_cut, seed):
            diploid_idx.append(j)
            if len(diploid_idx) >= max_normal:
                break
    preN = [cell_names[j] for j in diploid_idx]
    if len(diploid_idx) >= 3:
        basel = X[:, diploid_idx].mean(axis=1)
        warning = "ok"
    else:
        basel = X.mean(axis=1)
        warning = "no.confident.baseline"
    return BaselineResult(basel=basel, preN=preN, warning=warning,
                          labels=np.zeros(c, dtype=int))
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/baseline/gmm.py tests/unit/test_baseline_gmm.py
git commit -m "feat(baseline): per-cell GMM fallback mode"
```

---

## Task 4.4: `baseline/synthetic.py` — cell-line mode

Mirrors R `baseline.synthetic`.

**Files:**
- Create: `src/pycopykat/baseline/synthetic.py`
- Create: `tests/unit/test_baseline_synthetic.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
from pycopykat.baseline.synthetic import baseline_synthetic

def test_synthetic_returns_relative_matrix():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 60))
    res = baseline_synthetic(X, min_cells=10, seed=123)
    assert res.expr_relat.shape == X.shape
    assert res.labels.shape == (60,)
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

```python
# src/pycopykat/baseline/synthetic.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from pycopykat.baseline._shared import ward_cluster_with_min_size

@dataclass(slots=True)
class SyntheticBaselineResult:
    expr_relat: np.ndarray
    syn_norm: np.ndarray
    labels: np.ndarray

def baseline_synthetic(
    X: np.ndarray, *, min_cells: int = 10, seed: int = 123,
) -> SyntheticBaselineResult:
    """Pure cell-line mode (R baseline.synthetic).
    X : (genes, cells) smoothed matrix.
    """
    labels, _Z = ward_cluster_with_min_size(
        X.T, initial_k=6, min_cells=min_cells, distance="euclidean"
    )
    # Per-cluster per-gene std, then synthetic normal centered at 0
    rng = np.random.default_rng(seed)
    g, c = X.shape
    syn = np.zeros(g)
    expr_relat = np.empty_like(X)
    for lab in np.unique(labels):
        mask = labels == lab
        sd = X[:, mask].std(axis=1, ddof=1)
        syn_lab = rng.normal(0.0, sd, size=g)
        expr_relat[:, mask] = X[:, mask] - syn_lab[:, None]
        # accumulate any-cluster syn for return (matches R's behavior of last cluster)
        syn = syn_lab
    return SyntheticBaselineResult(expr_relat=expr_relat, syn_norm=syn, labels=labels)
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/baseline/synthetic.py tests/unit/test_baseline_synthetic.py
git commit -m "feat(baseline): synthetic cell-line mode"
```

---

# M5 — Segmentation & Bin Conversion

## Task 5.1: `segment/breakpoint.py` — KS-test breakpoint detection

**Files:**
- Create: `src/pycopykat/segment/breakpoint.py`
- Create: `tests/unit/test_segment_breakpoint.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
from pycopykat.segment.breakpoint import find_breakpoints

def test_detects_mean_shift():
    # Synthetic 200-long signal: mean 1 for [0:100), mean 5 for [100:200)
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.poisson(1, size=100), rng.poisson(5, size=100)]).astype(float)
    br = find_breakpoints(y, bins=25, ks_cut=0.1, seed=0)
    # Expect a breakpoint near 100
    assert any(abs(b - 100) <= 25 for b in br), f"breaks={br}"

def test_returns_endpoints():
    y = np.ones(200)
    br = find_breakpoints(y, bins=25, ks_cut=0.1, seed=0)
    assert br[0] == 0
    assert br[-1] == 199
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

```python
# src/pycopykat/segment/breakpoint.py
from __future__ import annotations
import numpy as np
from scipy.stats import ks_2samp
from pycopykat.kernels.mcmc_pg import pg_posterior_samples

def find_breakpoints(
    y: np.ndarray, *, bins: int, ks_cut: float, seed: int, mc: int = 1000,
) -> list[int]:
    """
    Detect breakpoints on a single 1D signal (consensus per cluster).
    Returns sorted list including 0 and n-1.
    """
    n = y.size
    if n < 3 * bins:
        return [0, n - 1]
    boundaries = list(range(0, (n // bins - 1) * bins + 1, bins))
    if boundaries[-1] != n - 1:
        boundaries.append(n - 1)
    breaks: list[int] = []
    for i in range(len(boundaries) - 2):
        s1, e1 = boundaries[i], boundaries[i + 1]
        s2, e2 = boundaries[i + 1] + 1, boundaries[i + 2]
        y1 = y[s1:e1]
        y2 = y[s2:e2]
        if y1.size == 0 or y2.size == 0:
            continue
        a1 = max(y1.mean(), 1e-3)
        a2 = max(y2.mean(), 1e-3)
        p1 = pg_posterior_samples(y1, alpha=a1, beta=1.0, mc=mc, seed=seed + i)
        p2 = pg_posterior_samples(y2, alpha=a2, beta=1.0, mc=mc, seed=seed + i + 10_000)
        stat, _ = ks_2samp(p1, p2)
        if stat > ks_cut:
            breaks.append(boundaries[i + 1])
    return sorted({0, *breaks, n - 1})
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/segment/breakpoint.py tests/unit/test_segment_breakpoint.py
git commit -m "feat(segment): KS-test breakpoint detection on Poisson-Gamma posterior"
```

---

## Task 5.2: `segment/mcmc.py` — segment-wise posterior mean per cell

**Files:**
- Create: `src/pycopykat/segment/mcmc.py`
- Create: `tests/unit/test_segment_mcmc.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
from pycopykat.segment.mcmc import segment_cells

def test_segment_assigns_segment_mean():
    rng = np.random.default_rng(0)
    # 50 cells, 200 genes
    fttmat = np.exp(rng.standard_normal((200, 50)) * 0.1)  # near 1
    clu = np.array([1] * 25 + [2] * 25)
    logCNA, BR = segment_cells(fttmat, clu, bins=25, ks_cut=0.2, seed=0, mc=200)
    assert logCNA.shape == fttmat.shape
    assert BR[0] == 0 and BR[-1] == fttmat.shape[0] - 1
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

```python
# src/pycopykat/segment/mcmc.py
from __future__ import annotations
import numpy as np
from joblib import Parallel, delayed
from pycopykat.kernels.mcmc_pg import pg_posterior_samples
from pycopykat.segment.breakpoint import find_breakpoints

def _consensus_per_cluster(fttmat: np.ndarray, clu: np.ndarray) -> np.ndarray:
    """Per-cluster gene-wise median, then exp() to mimic R's exp(CON)."""
    out = []
    for lab in np.unique(clu):
        med = np.median(fttmat[:, clu == lab], axis=1)
        out.append(np.exp(med))
    return np.column_stack(out)

def _union_breaks(consensus: np.ndarray, bins: int, ks_cut: float, seed: int, mc: int) -> list[int]:
    all_breaks: set[int] = {0, consensus.shape[0] - 1}
    for c in range(consensus.shape[1]):
        br = find_breakpoints(consensus[:, c], bins=bins, ks_cut=ks_cut,
                              seed=seed + 1000 * c, mc=mc)
        all_breaks.update(br)
    return sorted(all_breaks)

def _segment_one_cell(col: np.ndarray, BR: list[int], seed: int, mc: int) -> np.ndarray:
    x = np.empty_like(col)
    for i in range(len(BR) - 1):
        s, e = BR[i], BR[i + 1]
        seg = col[s : e + 1]
        a = max(seg.mean(), 1e-3)
        samp = pg_posterior_samples(seg, alpha=a, beta=1.0, mc=mc,
                                    seed=seed + i * 7919)
        x[s : e + 1] = samp.mean()
    return np.log(np.maximum(x, 1e-12))

def segment_cells(
    fttmat: np.ndarray, clu: np.ndarray, *,
    bins: int, ks_cut: float, seed: int, mc: int = 1000, n_jobs: int = 1,
) -> tuple[np.ndarray, list[int]]:
    """
    fttmat : (genes, cells) VST-smoothed relative expression (not yet exp'd).
    clu    : length-cells integer cluster labels.
    Returns (logCNA, BR).
    """
    consensus = _consensus_per_cluster(fttmat, clu)
    BR = _union_breaks(consensus, bins=bins, ks_cut=ks_cut, seed=seed, mc=mc)
    raw = np.exp(fttmat)
    cols = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_segment_one_cell)(raw[:, j], BR, seed + j, mc)
        for j in range(raw.shape[1])
    )
    logCNA = np.column_stack(cols)
    return logCNA, BR
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/segment/mcmc.py tests/unit/test_segment_mcmc.py
git commit -m "feat(segment): per-cell Poisson-Gamma segmentation with union breaks"
```

---

## Task 5.3: `cna/bins.py` — gene → 220KB genomic bin aggregation

**Files:**
- Create: `src/pycopykat/cna/bins.py`
- Create: `tests/unit/test_cna_bins.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pandas as pd
from pycopykat.cna.bins import aggregate_to_bins

def test_aggregate_medians_genes_within_bin():
    # Toy: 4 genes, 2 cells, 2 bins
    gene_anno = pd.DataFrame({
        "chromosome_name": [1, 1, 1, 1],
        "start_position": [100, 200, 300, 400],
        "end_position":   [150, 250, 350, 450],
        "abspos": [125, 225, 325, 425],
    })
    logCNA = np.array([[1.0, 2.0],
                       [2.0, 3.0],
                       [3.0, 4.0],
                       [4.0, 5.0]])
    bins = pd.DataFrame({
        "chrom": [1, 1],
        "abspos": [200, 400],
        "start": [100, 300],
        "end":   [300, 500],
    })
    out = aggregate_to_bins(logCNA, gene_anno, bins)
    # Bin 1 (100–300) → genes 0,1 → median of [1,2]=1.5 and [2,3]=2.5
    # Bin 2 (300–500) → genes 2,3 → median of [3,4]=3.5 and [4,5]=4.5
    np.testing.assert_allclose(out[0], [1.5, 2.5])
    np.testing.assert_allclose(out[1], [3.5, 4.5])
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

```python
# src/pycopykat/cna/bins.py
from __future__ import annotations
import numpy as np
import pandas as pd

def aggregate_to_bins(
    logCNA: np.ndarray, gene_anno: pd.DataFrame, bins: pd.DataFrame,
) -> np.ndarray:
    """
    logCNA    : (genes, cells) segmented log-expression
    gene_anno : genes × (chromosome_name, start_position, end_position)  (same order as logCNA rows)
    bins      : DataFrame with columns (chrom, start, end). One row per target bin.
    Returns (n_bins, cells) matrix of per-bin medians. Missing bins are forward-filled
    from nearest earlier non-missing bin; leading missing are back-filled.
    """
    centers = 0.5 * (gene_anno["start_position"].to_numpy() + gene_anno["end_position"].to_numpy())
    bin_chr = bins["chrom"].to_numpy()
    bin_start = bins["start"].to_numpy()
    bin_end = bins["end"].to_numpy()
    gene_chr = gene_anno["chromosome_name"].to_numpy()
    n_bins = len(bins)
    n_cells = logCNA.shape[1]
    out = np.full((n_bins, n_cells), np.nan)
    for b in range(n_bins):
        mask = (gene_chr == bin_chr[b]) & (centers >= bin_start[b]) & (centers < bin_end[b])
        if not mask.any():
            continue
        out[b] = np.median(logCNA[mask], axis=0)
    # Forward fill within each cell column
    last = np.full(n_cells, np.nan)
    for b in range(n_bins):
        missing = np.isnan(out[b])
        out[b, missing] = last[missing]
        good = ~np.isnan(out[b])
        last[good] = out[b, good]
    # Back-fill any remaining leading NaNs
    nxt = np.full(n_cells, np.nan)
    for b in range(n_bins - 1, -1, -1):
        missing = np.isnan(out[b])
        out[b, missing] = nxt[missing]
        good = ~np.isnan(out[b])
        nxt[good] = out[b, good]
    return out
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/cna/bins.py tests/unit/test_cna_bins.py
git commit -m "feat(cna): gene-to-220KB-bin median aggregation with fill"
```

---

## Task 5.4: Segment package `__init__.py` + integration test

- [ ] **Step 1: Write integration test combining segment + bins**

```python
# tests/unit/test_segment_bins_integration.py
import numpy as np
import pandas as pd
from pycopykat.segment.mcmc import segment_cells
from pycopykat.cna.bins import aggregate_to_bins

def test_segment_then_bin_roundtrip():
    rng = np.random.default_rng(0)
    g, c = 400, 20
    fttmat = rng.standard_normal((g, c)) * 0.1
    clu = np.array([1] * 10 + [2] * 10)
    logCNA, BR = segment_cells(fttmat, clu, bins=25, ks_cut=0.2, seed=0, mc=200)
    anno = pd.DataFrame({
        "chromosome_name": [1] * g,
        "start_position": np.arange(g) * 1000,
        "end_position": np.arange(g) * 1000 + 500,
        "abspos": np.arange(g) * 1000 + 250,
    })
    bins = pd.DataFrame({
        "chrom": [1] * 10,
        "start": np.arange(10) * 40_000,
        "end": (np.arange(10) + 1) * 40_000,
        "abspos": np.arange(10) * 40_000 + 20_000,
    })
    out = aggregate_to_bins(logCNA, anno, bins)
    assert out.shape == (10, c)
    assert not np.isnan(out).any()
```

- [ ] **Step 2: Run test → PASS (should pass since prior tasks done)**

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_segment_bins_integration.py
git commit -m "test: segment + bins integration"
```

---

# M6 — Classification, Subclone, Heatmap

## Task 6.1: `classify/predict.py` — diploid/aneuploid assignment

R copykat.R lines 345–399: Ward linkage on binned CNA matrix, k=2, assign "diploid" = cluster with max fraction of preN names.

**Files:**
- Create: `src/pycopykat/classify/predict.py`
- Create: `tests/unit/test_classify_predict.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pandas as pd
from pycopykat.classify.predict import predict_ploidy

def test_labels_known_normal_majority_cluster_as_diploid():
    rng = np.random.default_rng(0)
    # 20 cells, 50 bins
    cna = np.hstack([
        rng.normal(0, 0.05, size=(50, 10)),   # "diploid-like"
        rng.normal(0.3, 0.2, size=(50, 10)),  # "aneuploid-like"
    ])
    cells = [f"d{i}" for i in range(10)] + [f"a{i}" for i in range(10)]
    preN = [f"d{i}" for i in range(10)]
    pred = predict_ploidy(cna, cells, preN=preN, distance="euclidean")
    dip = set(pred.loc[pred["copykat.pred"] == "diploid", "cell"])
    assert len(dip & set(preN)) >= 8
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

```python
# src/pycopykat/classify/predict.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from pycopykat.kernels.distances import pdist_euclidean, pdist_pearson, pdist_spearman

_DIST = {"euclidean": pdist_euclidean, "pearson": pdist_pearson, "spearman": pdist_spearman}

def predict_ploidy(
    cna: np.ndarray, cells: list[str], *, preN: list[str], distance: str = "euclidean",
) -> pd.DataFrame:
    """
    cna : (bins, cells) adjusted CNA matrix.
    Returns DataFrame columns: cell, copykat.pred ('diploid'|'aneuploid').
    """
    d = _DIST[distance](cna.T)
    Z = linkage(d, method="ward")
    labels = fcluster(Z, t=2, criterion="maxclust")
    preN_set = set(preN)
    frac_normal = {
        lab: sum(cells[i] in preN_set for i, l in enumerate(labels) if l == lab)
             / max(1, (labels == lab).sum())
        for lab in np.unique(labels)
    }
    diploid_label = max(frac_normal, key=frac_normal.get)
    pred = ["diploid" if l == diploid_label else "aneuploid" for l in labels]
    return pd.DataFrame({"cell": cells, "copykat.pred": pred})
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/classify/predict.py tests/unit/test_classify_predict.py
git commit -m "feat(classify): diploid/aneuploid prediction via Ward k=2"
```

---

## Task 6.2: `classify/subclone.py` — dynamicTreeCut hybrid port

Port Langfelder et al. 2007 hybrid method. Simplified V1: iterative tree-cut starting from deepest branches, merging/splitting by cluster-size thresholds.

**Files:**
- Create: `src/pycopykat/classify/subclone.py`
- Create: `tests/unit/test_classify_subclone.py`

- [ ] **Step 1: Write failing test using 3 well-separated blobs**

```python
import numpy as np
from scipy.cluster.hierarchy import linkage
from pycopykat.classify.subclone import dynamic_tree_cut
from pycopykat.kernels.distances import pdist_euclidean

def test_finds_three_blobs():
    rng = np.random.default_rng(0)
    blobs = np.vstack([
        rng.normal(loc=0,  scale=0.05, size=(25, 20)),
        rng.normal(loc=5,  scale=0.05, size=(25, 20)),
        rng.normal(loc=10, scale=0.05, size=(25, 20)),
    ])
    Z = linkage(pdist_euclidean(blobs), method="ward")
    labels = dynamic_tree_cut(Z, min_cluster_size=10, deep_split=2)
    assert len(set(labels)) == 3
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement hybrid method**

Create `src/pycopykat/classify/subclone.py`:
```python
"""Port of WGCNA dynamicTreeCut (hybrid method).
Based on Langfelder, P., Zhang, B., & Horvath, S. (2008) Bioinformatics 24(5):719-720.
V1: simplified hybrid — identifies clusters by traversing merges in reverse height order,
greedily accepting clusters of >= min_cluster_size. Remaining unassigned leaves are labeled 0.
"""
from __future__ import annotations
import numpy as np

def _descendants(Z: np.ndarray, node: int, n: int) -> list[int]:
    """Return the original-leaf indices under a (possibly merged) node."""
    if node < n:
        return [node]
    left, right = int(Z[node - n, 0]), int(Z[node - n, 1])
    return _descendants(Z, left, n) + _descendants(Z, right, n)

def dynamic_tree_cut(
    Z: np.ndarray, *, min_cluster_size: int = 10, deep_split: int = 2,
) -> np.ndarray:
    """Assign each leaf a cluster label. Label 0 = unassigned (outlier).
    Greedy top-down: from the highest merge downward, accept any subtree
    of size >= min_cluster_size as its own cluster unless a child subtree
    already qualifies more tightly.
    deep_split (0..4) raises height sensitivity — larger = more splits.
    """
    n = Z.shape[0] + 1
    heights = Z[:, 2]
    max_h = heights.max()
    min_h = heights.min()
    # deep_split threshold: fraction of tree height below which we keep splitting.
    cut_h = min_h + (max_h - min_h) * (1 - 0.15 * (deep_split + 1))
    labels = np.zeros(n, dtype=int)
    next_label = 1

    def assign(node: int, current_cut: float) -> None:
        nonlocal next_label
        leaves = _descendants(Z, node, n)
        if len(leaves) < min_cluster_size:
            return
        if node < n:
            return
        merge_h = Z[node - n, 2]
        left, right = int(Z[node - n, 0]), int(Z[node - n, 1])
        l_size = len(_descendants(Z, left, n))
        r_size = len(_descendants(Z, right, n))
        # If both sides are big enough and we're still above cut, descend.
        if merge_h > current_cut and l_size >= min_cluster_size and r_size >= min_cluster_size:
            assign(left, current_cut)
            assign(right, current_cut)
            return
        # Otherwise accept this subtree as one cluster
        for leaf in leaves:
            if labels[leaf] == 0:
                labels[leaf] = next_label
        next_label += 1

    root = n + Z.shape[0] - 1
    assign(root, cut_h)
    return labels
```

- [ ] **Step 4: Run test → PASS**

If recursion depth exceeds limits on large trees, convert `_descendants` and `assign` to iterative. Cache `_descendants` results in a dict keyed by node.

- [ ] **Step 5: Add comparison test vs R dynamicTreeCut (fuzzier assertion)**

Ensure R has dynamicTreeCut installed:
```bash
Rscript -e 'if (!requireNamespace("dynamicTreeCut", quietly=TRUE)) install.packages("dynamicTreeCut", repos="https://cloud.r-project.org")'
```

Append to `test_classify_subclone.py`:
```python
import subprocess

def test_matches_r_dynamictreecut_loosely(tmp_path):
    rng = np.random.default_rng(1)
    blobs = np.vstack([
        rng.normal(loc=0,  scale=0.1, size=(30, 15)),
        rng.normal(loc=3,  scale=0.1, size=(30, 15)),
        rng.normal(loc=6,  scale=0.1, size=(30, 15)),
    ])
    np.savetxt(tmp_path / "X.txt", blobs)
    rscript = tmp_path / "run.R"
    rscript.write_text(f"""
      suppressMessages(library(dynamicTreeCut))
      X <- as.matrix(read.table("{tmp_path}/X.txt"))
      d <- dist(X, method="euclidean")
      h <- hclust(d, method="ward.D2")
      lab <- cutreeDynamic(h, distM=as.matrix(d), deepSplit=2, minClusterSize=10)
      write.table(lab, "{tmp_path}/lab.txt", row.names=FALSE, col.names=FALSE)
    """)
    subprocess.run(["Rscript", str(rscript)], check=True, capture_output=True)
    r_lab = np.loadtxt(tmp_path / "lab.txt").astype(int)

    Z = linkage(pdist_euclidean(blobs), method="ward")
    py_lab = dynamic_tree_cut(Z, min_cluster_size=10, deep_split=2)
    from sklearn.metrics import fowlkes_mallows_score
    fmi = fowlkes_mallows_score(r_lab, py_lab)
    assert fmi >= 0.70, f"FMI too low: {fmi:.3f}"
```

- [ ] **Step 6: Run test; if FMI < 0.70, document divergence**

If FMI < 0.70 on this toy case, we invoke the fallback from spec §10.1: downgrade subclone acceptance to "count ± 2, FMI ≥ 0.70" and note in RELEASE_NOTES.

- [ ] **Step 7: Commit**

```bash
git add src/pycopykat/classify/subclone.py tests/unit/test_classify_subclone.py
git commit -m "feat(classify): port dynamicTreeCut hybrid method for subclone detection"
```

---

## Task 6.3: `viz/heatmap.py` — matplotlib heatmap with chromosome side bar

**Files:**
- Create: `src/pycopykat/viz/heatmap.py`
- Create: `tests/unit/test_viz_heatmap.py`

- [ ] **Step 1: Write failing test**

```python
from pathlib import Path
import numpy as np
import pandas as pd
from pycopykat.viz.heatmap import plot_cna_heatmap

def test_writes_png(tmp_path):
    cna = pd.DataFrame(
        np.random.default_rng(0).standard_normal((200, 30)),
        index=pd.MultiIndex.from_arrays(
            [[1]*100 + [2]*100, np.arange(200)], names=["chrom", "bin"]
        ),
        columns=[f"c{i}" for i in range(30)],
    )
    pred = pd.DataFrame({
        "cell": cna.columns,
        "copykat.pred": ["aneuploid" if i % 2 else "diploid" for i in range(30)],
    })
    out = tmp_path / "heatmap.png"
    plot_cna_heatmap(cna, pred, output=out)
    assert out.exists() and out.stat().st_size > 5_000
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

```python
# src/pycopykat/viz/heatmap.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def plot_cna_heatmap(
    cna: pd.DataFrame, prediction: pd.DataFrame, output: Path,
    *, vmin: float = -1.0, vmax: float = 1.0,
) -> None:
    """Simple matplotlib heatmap; rows=cells (grouped by prediction), cols=bins."""
    pred_map = dict(zip(prediction["cell"], prediction["copykat.pred"]))
    col_order = sorted(cna.columns, key=lambda c: (pred_map.get(c, "x"), c))
    M = cna[col_order].to_numpy().T  # cells × bins
    chrom = cna.index.get_level_values(0).to_numpy() \
        if isinstance(cna.index, pd.MultiIndex) else np.zeros(M.shape[1])

    fig, (ax_cb, ax_heat) = plt.subplots(
        2, 1, figsize=(12, 10),
        gridspec_kw={"height_ratios": [0.5, 20], "hspace": 0.02},
    )
    # Chromosome bar
    uniq = np.unique(chrom)
    cmap_chr = plt.get_cmap("tab20")(np.linspace(0, 1, len(uniq)))
    chr_colors = np.zeros((1, M.shape[1], 4))
    for k, u in enumerate(uniq):
        chr_colors[0, chrom == u] = cmap_chr[k]
    ax_cb.imshow(chr_colors, aspect="auto", interpolation="nearest")
    ax_cb.set_yticks([])
    ax_cb.set_xticks([])
    # Main heatmap
    ax_heat.imshow(M, aspect="auto", cmap="RdBu_r",
                   norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax))
    ax_heat.set_ylabel(f"{M.shape[0]} cells")
    ax_heat.set_xlabel(f"{M.shape[1]} bins")
    ax_heat.set_yticks([])
    ax_heat.set_xticks([])
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/viz/heatmap.py tests/unit/test_viz_heatmap.py
git commit -m "feat(viz): matplotlib CNA heatmap with chromosome side bar"
```

---

## Task 6.4: Baseline adjustment helper (pre-classification)

R copykat.R lines 371–391: center, per-bin sd on diploid cluster, threshold-mute small deviations, recenter, re-cluster.

**Files:**
- Create: `src/pycopykat/classify/adjust_pipeline.py`
- Create: `tests/unit/test_classify_adjust_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
from pycopykat.classify.adjust_pipeline import baseline_adjust

def test_adjust_mutes_values_within_sd_window():
    rng = np.random.default_rng(0)
    cna = rng.standard_normal((100, 20)) * 0.1
    diploid_mask = np.zeros(20, dtype=bool)
    diploid_mask[:10] = True
    out = baseline_adjust(cna, diploid_mask, factor=0.25)
    # After adjust, diploid columns should have smaller range
    assert out[:, diploid_mask].std() < cna[:, diploid_mask].std()
    assert out.shape == cna.shape
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

```python
# src/pycopykat/classify/adjust_pipeline.py
from __future__ import annotations
import numpy as np
from pycopykat.kernels.adjust import adjust_threshold

def baseline_adjust(cna: np.ndarray, diploid_mask: np.ndarray, *, factor: float = 0.25) -> np.ndarray:
    """
    cna : (bins, cells) CNA matrix.
    diploid_mask : length-cells bool, selecting reference diploid cells.
    Subtracts diploid-cluster mean, centers columns, computes per-bin sd on
    diploid cluster, mutes small deviations via adjust_threshold, re-centers.
    """
    assert diploid_mask.sum() >= 2, "need >=2 diploid cells for sd"
    dip_mean = cna[:, diploid_mask].mean(axis=1, keepdims=True)
    rel = cna - dip_mean
    rel -= rel.mean(axis=0, keepdims=True)
    sd = rel[:, diploid_mask].std(axis=1, ddof=1)
    base = rel[:, diploid_mask].mean(axis=1)
    out = adjust_threshold(rel, base=base, sd=sd, factor=factor)
    out -= out.mean(axis=0, keepdims=True)
    return out
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/classify/adjust_pipeline.py tests/unit/test_classify_adjust_pipeline.py
git commit -m "feat(classify): baseline threshold adjustment before final clustering"
```

---

# M7 — Pipeline Orchestration, CLI, Regression

## Task 7.1: `pipeline.py` — top-level `copykat()` orchestration

**Files:**
- Create: `src/pycopykat/pipeline.py`
- Modify: `src/pycopykat/__init__.py`
- Create: `tests/unit/test_pipeline_smoke.py`

- [ ] **Step 1: Write smoke test on tiny synthetic data**

```python
import numpy as np
import pandas as pd
from pycopykat import copykat
from pycopykat.config import CopykatConfig

def test_copykat_runs_on_tiny_data():
    rng = np.random.default_rng(0)
    from pycopykat.io.annotation import load_hg20_annotation
    ann = load_hg20_annotation().head(500)
    genes = ann["hgnc_symbol"].dropna().unique().tolist()[:300]
    mat = pd.DataFrame(rng.poisson(3, size=(len(genes), 50)).astype(int),
                       index=genes, columns=[f"c{i}" for i in range(50)])
    cfg = CopykatConfig(min_gene_per_cell=10, low_dr=0.01, up_dr=0.02,
                        win_size=10, ks_cut=0.3, n_jobs=1, sam_name="tiny")
    res = copykat(mat, config=cfg)
    assert res.prediction.shape[0] >= 10
    assert set(res.prediction["copykat.pred"]).issubset({"diploid", "aneuploid"})
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement orchestration**

Create `src/pycopykat/pipeline.py`:
```python
from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from pycopykat.config import CopykatConfig
from pycopykat.result import CopykatResult
from pycopykat.io.annotation import annotate_genes, load_hg20_bins
from pycopykat.preprocess.filter import (
    filter_cells_and_genes, filter_cells_by_chrom_coverage,
)
from pycopykat.preprocess.normalize import vst_center
from pycopykat.preprocess.smooth import smooth_cells
from pycopykat.baseline.auto import baseline_norm_cl
from pycopykat.baseline.gmm import baseline_gmm
from pycopykat.baseline.synthetic import baseline_synthetic
from pycopykat.segment.mcmc import segment_cells
from pycopykat.cna.bins import aggregate_to_bins
from pycopykat.classify.adjust_pipeline import baseline_adjust
from pycopykat.classify.predict import predict_ploidy
from pycopykat.classify.subclone import dynamic_tree_cut
from pycopykat.kernels.distances import (
    pdist_euclidean, pdist_pearson, pdist_spearman,
)
from scipy.cluster.hierarchy import linkage

log = logging.getLogger("pycopykat")

_DIST = {"euclidean": pdist_euclidean, "pearson": pdist_pearson, "spearman": pdist_spearman}

def copykat(
    mat: pd.DataFrame | np.ndarray,
    config: CopykatConfig | None = None,
    **kwargs,
) -> CopykatResult:
    """Full CopyKAT pipeline. `mat` is genes × cells (DataFrame preferred)."""
    cfg = config or CopykatConfig(**kwargs)
    warnings: list[str] = []
    log.info("step 1: filter")
    if not isinstance(mat, pd.DataFrame):
        raise TypeError("pass a pandas DataFrame with gene names in the index")
    mat1, stat1 = filter_cells_and_genes(
        mat, min_gene_per_cell=cfg.min_gene_per_cell, low_dr=cfg.low_dr
    )
    if stat1["data_quality"] == "low":
        warnings.append("low data quality — UP.DR set to LOW.DR")
        up_dr = cfg.low_dr
    else:
        up_dr = cfg.up_dr

    log.info("step 2: annotate")
    annotated = annotate_genes(mat1, id_type=cfg.id_type, genome=cfg.genome)
    cell_cols = [c for c in annotated.columns
                 if c not in {"hgnc_symbol", "ensembl_gene_id",
                              "chromosome_name", "chrom",
                              "start_position", "end_position", "abspos"}]
    gene_anno = annotated[["hgnc_symbol", "chromosome_name", "start_position",
                           "end_position", "abspos"]].copy()
    X0 = annotated[cell_cols].to_numpy(dtype=np.float64)

    log.info("step 3: normalize + smooth")
    X_norm = vst_center(X0)
    X_smooth = smooth_cells(X_norm)

    # Note: R copykat applies the UP.DR and chrom-coverage filters AFTER
    # smoothing (copykat.R lines 184-217). Order here must match R; if the
    # regression test in Task 7.5 shows divergence in retained gene count,
    # re-verify against the R source before altering other steps.
    der = (X0 > 0).sum(axis=1) / X0.shape[1]
    keep_g = der > up_dr
    X_smooth = X_smooth[keep_g]
    gene_anno = gene_anno.loc[keep_g].reset_index(drop=True)
    chrom_int = gene_anno["chromosome_name"].astype(str).str.replace("X", "23").str.replace("Y", "24").astype(int).to_numpy()
    X_smooth2, dropped = filter_cells_by_chrom_coverage(
        X_smooth, chrom_int, ngene_chr=cfg.ngene_chr
    )
    cells = [c for i, c in enumerate(cell_cols) if i not in set(dropped)]
    if len(cells) < 20:
        raise ValueError("too few cells remain after chromosome-coverage filter")

    log.info("step 4: baseline")
    if cfg.cell_line:
        synres = baseline_synthetic(X_smooth2, min_cells=10, seed=123)
        X_rel = synres.expr_relat
        preN: list[str] = []
        warnings.append("cell-line mode")
    elif cfg.norm_cell_names:
        basel = np.median(X_smooth2[:, [i for i, c in enumerate(cells) if c in set(cfg.norm_cell_names)]], axis=1)
        X_rel = X_smooth2 - basel[:, None]
        preN = [c for c in cells if c in set(cfg.norm_cell_names)]
    else:
        br = baseline_norm_cl(X_smooth2, cell_names=cells,
                              min_cells=5, n_jobs=cfg.n_jobs, seed=cfg.seed)
        warnings.append(br.warning)
        if br.warning == "unclassified.prediction":
            log.info("fallback → baseline.GMM")
            br = baseline_gmm(X_smooth2, cells, seed=cfg.seed)
            warnings.append(br.warning)
        basel = br.basel
        X_rel = X_smooth2 - basel[:, None]
        preN = br.preN

    log.info("step 5: segment")
    # Use ward clustering labels for segmentation consensus
    from scipy.cluster.hierarchy import fcluster
    d = _DIST[cfg.distance](X_rel.T)
    Z_pre = linkage(d, method="ward")
    clu = fcluster(Z_pre, t=min(6, X_rel.shape[1] // 5), criterion="maxclust")
    logCNA, BR = segment_cells(X_rel, clu, bins=cfg.win_size, ks_cut=cfg.ks_cut,
                               seed=cfg.seed, mc=1000, n_jobs=cfg.n_jobs)

    log.info("step 6: bin to 220KB")
    bins_df = load_hg20_bins()
    # Rename cols to match aggregate_to_bins contract
    bins_df = bins_df.rename(columns={"chrompos": "start"}) if "chrompos" in bins_df.columns else bins_df
    if "start" not in bins_df.columns:
        # derive start/end from chrom + abspos + 220000 resolution
        bins_df = bins_df.sort_values(["chrom", "abspos"]).copy()
        bins_df["start"] = bins_df["abspos"] - 110_000
        bins_df["end"] = bins_df["abspos"] + 110_000
    cna_bins = aggregate_to_bins(logCNA, gene_anno, bins_df)

    log.info("step 7: adjust + classify")
    preN_set = set(preN) if preN else set(cells[: max(5, len(cells) // 10)])
    diploid_mask = np.array([c in preN_set for c in cells])
    if diploid_mask.sum() < 2:
        diploid_mask[:2] = True  # floor
    cna_adj = baseline_adjust(cna_bins, diploid_mask, factor=0.25)
    pred = predict_ploidy(cna_adj, cells, preN=list(preN_set),
                          distance=cfg.distance)

    log.info("step 8: subclone")
    aneu = pred["copykat.pred"] == "aneuploid"
    aneu_cells = pred.loc[aneu, "cell"].tolist()
    if len(aneu_cells) >= 20:
        cna_aneu = cna_adj[:, [i for i, c in enumerate(cells) if c in set(aneu_cells)]]
        Zs = linkage(_DIST[cfg.distance](cna_aneu.T), method="ward")
        sub_labels = dynamic_tree_cut(Zs, min_cluster_size=max(10, len(aneu_cells) // 20),
                                      deep_split=2)
        subclone = pd.Series(sub_labels, index=aneu_cells, name="subclone")
    else:
        subclone = pd.Series(dtype=int, name="subclone")

    cna_df = pd.DataFrame(
        cna_adj,
        index=pd.MultiIndex.from_frame(bins_df[["chrom", "start", "end"]]),
        columns=cells,
    )
    Z_final = linkage(_DIST[cfg.distance](cna_adj.T), method="ward")
    return CopykatResult(
        cna_mat=cna_df, prediction=pred, linkage=Z_final,
        subclone=subclone, warnings=tuple(warnings),
    )
```

- [ ] **Step 4: Export from `__init__.py`**

Overwrite `src/pycopykat/__init__.py`:
```python
from pycopykat.pipeline import copykat
from pycopykat.config import CopykatConfig
from pycopykat.result import CopykatResult

__all__ = ["copykat", "CopykatConfig", "CopykatResult"]
__version__ = "0.1.0.dev0"
```

- [ ] **Step 5: Run test → PASS (smoke only)**

```bash
uv run pytest tests/unit/test_pipeline_smoke.py -v -s
```

- [ ] **Step 6: Commit**

```bash
git add src/pycopykat/pipeline.py src/pycopykat/__init__.py tests/unit/test_pipeline_smoke.py
git commit -m "feat(pipeline): top-level copykat() orchestration"
```

---

## Task 7.2: `cli.py` — typer CLI entry point

**Files:**
- Create: `src/pycopykat/cli.py`
- Create: `tests/unit/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
from pathlib import Path
from typer.testing import CliRunner
from pycopykat.cli import app

runner = CliRunner()

def test_cli_help():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "run" in res.output

def test_cli_run_on_csv(tmp_path):
    # Build tiny CSV
    import pandas as pd
    from pycopykat.io.annotation import load_hg20_annotation
    import numpy as np
    ann = load_hg20_annotation().head(400)
    genes = ann["hgnc_symbol"].dropna().unique().tolist()[:200]
    mat = pd.DataFrame(np.random.default_rng(0).poisson(3, size=(len(genes), 40)),
                       index=genes, columns=[f"c{i}" for i in range(40)])
    csv = tmp_path / "in.csv"
    mat.to_csv(csv)
    out = tmp_path / "out"
    res = runner.invoke(app, [
        "run", "--input", str(csv), "--output-dir", str(out),
        "--sam-name", "cli", "--min-gene-per-cell", "5",
        "--low-dr", "0.01", "--up-dr", "0.02",
        "--win-size", "10", "--ks-cut", "0.3",
    ])
    assert res.exit_code == 0, res.output
    assert (out / "cli_copykat_prediction.txt").exists()
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement CLI**

```python
# src/pycopykat/cli.py
from __future__ import annotations
from pathlib import Path
import typer
import pandas as pd
from pycopykat import copykat, CopykatConfig

app = typer.Typer(help="pycopykat — Python rewrite of CopyKAT")

@app.command()
def run(
    input: Path = typer.Option(..., help="genes×cells CSV/TSV/h5ad"),
    output_dir: Path = typer.Option(Path("."), help="output directory"),
    sam_name: str = typer.Option("sample"),
    id_type: str = typer.Option("Symbol"),
    genome: str = typer.Option("hg20"),
    cell_line: bool = typer.Option(False),
    ngene_chr: int = typer.Option(5),
    min_gene_per_cell: int = typer.Option(200),
    low_dr: float = typer.Option(0.05),
    up_dr: float = typer.Option(0.1),
    win_size: int = typer.Option(25),
    ks_cut: float = typer.Option(0.1),
    distance: str = typer.Option("euclidean"),
    n_jobs: int = typer.Option(1),
    seed: int = typer.Option(1234),
    output_h5ad: bool = typer.Option(False),
) -> None:
    sep = "\t" if input.suffix in (".tsv", ".txt") else ","
    if input.suffix == ".h5ad":
        import anndata as ad
        ad_obj = ad.read_h5ad(input)
        mat = pd.DataFrame(ad_obj.X.T if hasattr(ad_obj.X, "T") else ad_obj.X.T,
                           index=ad_obj.var_names, columns=ad_obj.obs_names)
    else:
        mat = pd.read_csv(input, sep=sep, index_col=0)
    cfg = CopykatConfig(
        id_type=id_type, genome=genome, cell_line=cell_line,
        ngene_chr=ngene_chr, min_gene_per_cell=min_gene_per_cell,
        low_dr=low_dr, up_dr=up_dr, win_size=win_size, ks_cut=ks_cut,
        distance=distance, n_jobs=n_jobs, seed=seed, sam_name=sam_name,
        output_dir=output_dir, output_h5ad=output_h5ad,
    )
    res = copykat(mat, config=cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    res.to_txt(output_dir, sam_name)
    if output_h5ad:
        res.to_h5ad(output_dir / f"{sam_name}_copykat.h5ad")
    typer.echo(f"wrote outputs under {output_dir}")
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/cli.py tests/unit/test_cli.py
git commit -m "feat(cli): pycopykat run command"
```

---

## Task 7.3: `validation/r_runner.py` — call R copykat via subprocess

**Files:**
- Create: `src/pycopykat/validation/r_runner.py`
- Create: `tests/unit/test_r_runner.py`

- [ ] **Step 1: Write failing test**

```python
from pathlib import Path
import numpy as np
import pandas as pd
from pycopykat.validation.r_runner import run_r_copykat

def test_run_r_copykat_on_bundled_data(tmp_path):
    # Load exp.rawdata from the R package
    rda = Path("/media/jason/T7/rerbulid/copykat-R/data/exp.rawdata.rda")
    import pyreadr
    res = pyreadr.read_r(str(rda))
    mat = next(iter(res.values()))  # DataFrame, genes × cells
    # Subsample for speed
    mat = mat.iloc[:3000, :100]
    outdir = tmp_path / "r_out"
    out = run_r_copykat(mat, outdir, sam_name="rtest", n_cores=4,
                       min_gene_per_cell=50, low_dr=0.01, up_dr=0.02)
    assert (out / "rtest_copykat_prediction.txt").exists()
    pred = pd.read_csv(out / "rtest_copykat_prediction.txt", sep="\t")
    assert "copykat.pred" in pred.columns
```

Note: this test runs R copykat on real data — expect 1-5 min runtime. Mark it slow.

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

```python
# src/pycopykat/validation/r_runner.py
from __future__ import annotations
from pathlib import Path
import subprocess
import tempfile
import pandas as pd
import pyreadr

R_TEMPLATE = r"""
suppressPackageStartupMessages({{
  library(copykat)
}})
mat <- read.table("{input_tsv}", header=TRUE, row.names=1, sep="\t", check.names=FALSE)
setwd("{workdir}")
set.seed(1234)
res <- copykat::copykat(
  rawmat = as.matrix(mat),
  id.type = "{id_type}",
  sam.name = "{sam_name}",
  n.cores = {n_cores},
  ngene.chr = {ngene_chr},
  min.gene.per.cell = {min_gene_per_cell},
  LOW.DR = {low_dr},
  UP.DR = {up_dr},
  win.size = {win_size},
  KS.cut = {ks_cut},
  distance = "{distance}",
  plot.genes = "FALSE"
)
saveRDS(res, "{rds_path}")
cat("R copykat completed\n")
"""

def run_r_copykat(
    mat: pd.DataFrame, output_dir: Path, *, sam_name: str = "rref",
    id_type: str = "S", n_cores: int = 4, ngene_chr: int = 5,
    min_gene_per_cell: int = 200, low_dr: float = 0.05, up_dr: float = 0.1,
    win_size: int = 25, ks_cut: float = 0.1, distance: str = "euclidean",
) -> Path:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv = output_dir / "input.tsv"
    mat.to_csv(tsv, sep="\t")
    rds = output_dir / "result.rds"
    script = output_dir / "run_r.R"
    script.write_text(R_TEMPLATE.format(
        input_tsv=tsv, workdir=output_dir, sam_name=sam_name, id_type=id_type,
        n_cores=n_cores, ngene_chr=ngene_chr, min_gene_per_cell=min_gene_per_cell,
        low_dr=low_dr, up_dr=up_dr, win_size=win_size, ks_cut=ks_cut,
        distance=distance, rds_path=rds,
    ))
    subprocess.run(["Rscript", str(script)], check=True, capture_output=False)
    return output_dir

def load_r_prediction(output_dir: Path, sam_name: str) -> pd.DataFrame:
    return pd.read_csv(output_dir / f"{sam_name}_copykat_prediction.txt", sep="\t")

def load_r_cna(output_dir: Path, sam_name: str) -> pd.DataFrame:
    return pd.read_csv(output_dir / f"{sam_name}_copykat_CNA_results.txt", sep="\t")
```

- [ ] **Step 4: Run test → PASS**

Expect wall time 1-3 minutes. If failing with R library path issues, verify `.libPaths()` in R matches `/home/jason/R/x86_64-pc-linux-gnu-library/4.5`.

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/validation/r_runner.py tests/unit/test_r_runner.py
git commit -m "feat(validation): subprocess wrapper to run R copykat for reference outputs"
```

---

## Task 7.4: `validation/metrics.py` — ARI / κ / Spearman / FMI

**Files:**
- Create: `src/pycopykat/validation/metrics.py`
- Create: `tests/unit/test_validation_metrics.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
import pandas as pd
from pycopykat.validation.metrics import compare_predictions, compare_cna

def test_compare_predictions():
    pred_r = pd.DataFrame({"cell": [f"c{i}" for i in range(20)],
                           "copykat.pred": ["diploid"]*10 + ["aneuploid"]*10})
    pred_py = pd.DataFrame({"cell": [f"c{i}" for i in range(20)],
                            "copykat.pred": ["diploid"]*9 + ["aneuploid"]*11})
    m = compare_predictions(pred_r, pred_py)
    assert m["n_shared"] == 20
    assert m["ari"] > 0.8
    assert 0.0 <= m["kappa"] <= 1.0

def test_compare_cna():
    rng = np.random.default_rng(0)
    cna_r = pd.DataFrame(rng.standard_normal((100, 10)),
                         columns=[f"c{i}" for i in range(10)])
    cna_py = cna_r + rng.standard_normal(cna_r.shape) * 0.05
    m = compare_cna(cna_r, cna_py, method="spearman")
    assert m["median_r"] > 0.9
```

- [ ] **Step 2: Run test → FAIL**

- [ ] **Step 3: Implement**

```python
# src/pycopykat/validation/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, fowlkes_mallows_score
from scipy.stats import spearmanr, pearsonr

def compare_predictions(pred_r: pd.DataFrame, pred_py: pd.DataFrame) -> dict:
    m = pred_r.merge(pred_py, on="cell", suffixes=("_r", "_py"))
    return {
        "n_shared": len(m),
        "ari": adjusted_rand_score(m["copykat.pred_r"], m["copykat.pred_py"]),
        "kappa": cohen_kappa_score(m["copykat.pred_r"], m["copykat.pred_py"]),
        "fmi": fowlkes_mallows_score(
            (m["copykat.pred_r"] == "aneuploid").astype(int),
            (m["copykat.pred_py"] == "aneuploid").astype(int),
        ),
    }

def compare_cna(cna_r: pd.DataFrame, cna_py: pd.DataFrame, *, method: str = "spearman") -> dict:
    shared = sorted(set(cna_r.columns) & set(cna_py.columns))
    if not shared:
        return {"n_shared": 0, "median_r": float("nan")}
    # Align rows by min-length (if bin sets differ, assume same order / crop)
    n = min(len(cna_r), len(cna_py))
    rs = []
    fn = spearmanr if method == "spearman" else pearsonr
    for c in shared:
        r, _ = fn(cna_r[c].to_numpy()[:n], cna_py[c].to_numpy()[:n])
        rs.append(r)
    return {
        "n_shared": len(shared), "method": method,
        "median_r": float(np.median(rs)), "mean_r": float(np.mean(rs)),
        "min_r": float(np.min(rs)),
    }
```

- [ ] **Step 4: Run test → PASS**

- [ ] **Step 5: Commit**

```bash
git add src/pycopykat/validation/metrics.py tests/unit/test_validation_metrics.py
git commit -m "feat(validation): ARI/kappa/FMI/Spearman comparison metrics"
```

---

## Task 7.5: `test_regression.py` — end-to-end on `exp.rawdata`

**Files:**
- Create: `tests/test_regression.py`

- [ ] **Step 1: Write regression test**

```python
"""Strict regression: Python version vs installed R copykat on exp.rawdata.rda."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadr
import pytest
from pycopykat import copykat, CopykatConfig
from pycopykat.validation.r_runner import run_r_copykat, load_r_prediction, load_r_cna
from pycopykat.validation.metrics import compare_predictions, compare_cna

REF_RDA = Path("/media/jason/T7/rerbulid/copykat-R/data/exp.rawdata.rda")

@pytest.mark.slow
def test_regression_exp_rawdata(tmp_path):
    mat = next(iter(pyreadr.read_r(str(REF_RDA)).values()))
    # Full data
    r_out = tmp_path / "r"
    run_r_copykat(mat, r_out, sam_name="reg", n_cores=4)
    pred_r = load_r_prediction(r_out, "reg")
    cna_r = load_r_cna(r_out, "reg")

    cfg = CopykatConfig(n_jobs=4, sam_name="py", output_dir=tmp_path / "py")
    res_py = copykat(mat, config=cfg)
    pred_py = res_py.prediction
    cna_py = res_py.cna_mat.reset_index()

    pred_cmp = compare_predictions(pred_r, pred_py)
    cna_cmp = compare_cna(
        cna_r.set_index(cna_r.columns[:3].tolist()),
        res_py.cna_mat, method="spearman",
    )
    print(pred_cmp, cna_cmp)
    assert pred_cmp["ari"] >= 0.90, f"ARI {pred_cmp['ari']:.3f} < 0.90"
    assert pred_cmp["kappa"] >= 0.90, f"kappa {pred_cmp['kappa']:.3f} < 0.90"
    assert cna_cmp["median_r"] >= 0.95, f"Spearman median {cna_cmp['median_r']:.3f} < 0.95"
```

- [ ] **Step 2: Run test → likely FAIL first time**

```bash
uv run pytest tests/test_regression.py -v -s -m slow
```

Expected at first: may fail on thresholds. That's OK — this test **drives** the debugging loop of the whole M7 milestone. Work through each failing metric:
- Low ARI → baseline step misidentifies diploid cells. Check `baseline_norm_cl` diagnostics.
- Low Spearman → segmentation diverges. Compare `BR` (breakpoints) count between R and Py; inspect `find_breakpoints` logic.

- [ ] **Step 3: Commit once green**

```bash
git add tests/test_regression.py
git commit -m "test: end-to-end regression on exp.rawdata matching R reference"
```

---

# M8 — 3CA External Validation

## Task 8.1: Dataset fetcher script

**Files:**
- Create: `scripts/fetch_3ca.py`

- [ ] **Step 1: Implement fetcher (manual step, interactive)**

```python
# scripts/fetch_3ca.py
"""Download three 3CA datasets for validation.
Visit https://www.weizmann.ac.il/sites/3CA/ and download:
  1. Gao2021_TNBC (Breast — TNBC subset)
  2. Peng2019_PDAC (Pancreas)
  3. Tirosh2016_Melanoma

Save each as a tab-separated genes × cells matrix to data/3ca/.
"""
from pathlib import Path
import sys

TARGET = Path(__file__).resolve().parents[1] / "data" / "3ca"

def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    print("Download manually from https://www.weizmann.ac.il/sites/3CA/")
    print(f"Place files under {TARGET}/:")
    print("  Gao2021_TNBC/counts.tsv")
    print("  Peng2019_PDAC/counts.tsv")
    print("  Tirosh2016_Melanoma/counts.tsv")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and download manually**

```bash
uv run python scripts/fetch_3ca.py
# Then manually download from 3CA portal
```

- [ ] **Step 3: Commit**

```bash
git add scripts/fetch_3ca.py
git commit -m "scripts: 3CA dataset fetcher placeholder"
```

---

## Task 8.2: `tests/test_3ca.py` — smoke validation on 3CA data

**Files:**
- Create: `tests/test_3ca.py`

- [ ] **Step 1: Write parameterized test**

```python
import os
from pathlib import Path
import pandas as pd
import pytest
from pycopykat import copykat, CopykatConfig
from pycopykat.validation.r_runner import run_r_copykat, load_r_prediction, load_r_cna
from pycopykat.validation.metrics import compare_predictions, compare_cna

DATA = Path(__file__).resolve().parents[1] / "data" / "3ca"

DATASETS = [
    ("Gao2021_TNBC",       "tnbc"),
    ("Peng2019_PDAC",      "pdac"),
    ("Tirosh2016_Melanoma","mel"),
]

@pytest.mark.slow
@pytest.mark.parametrize("folder, sam", DATASETS)
def test_3ca_dataset(folder, sam, tmp_path):
    tsv = DATA / folder / "counts.tsv"
    if not tsv.exists():
        pytest.skip(f"{tsv} missing; run scripts/fetch_3ca.py")
    mat = pd.read_csv(tsv, sep="\t", index_col=0)
    r_dir = tmp_path / "r"
    run_r_copykat(mat, r_dir, sam_name=sam, n_cores=4)
    pred_r = load_r_prediction(r_dir, sam)
    cna_r = load_r_cna(r_dir, sam)

    cfg = CopykatConfig(n_jobs=4, sam_name=sam, output_dir=tmp_path / "py")
    res = copykat(mat, config=cfg)
    p = compare_predictions(pred_r, res.prediction)
    c = compare_cna(cna_r.set_index(cna_r.columns[:3].tolist()), res.cna_mat, method="spearman")
    print(folder, p, c)
    assert p["ari"] >= 0.80, f"{folder} ARI {p['ari']:.3f} < 0.80"
    assert c["median_r"] >= 0.90, f"{folder} Spearman {c['median_r']:.3f} < 0.90"
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/test_3ca.py -v -s -m slow
```

Acceptance: 2 of 3 pass; the failing one must have a root-cause write-up in `docs/validation_notes.md`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_3ca.py
git commit -m "test: 3CA external validation suite"
```

---

## Task 8.3: Validation report generator

**Files:**
- Create: `scripts/validation_report.py`

- [ ] **Step 1: Implement**

```python
# scripts/validation_report.py
"""Run validation on all 3CA datasets and produce a markdown report."""
from pathlib import Path
import json
import pandas as pd
from pycopykat import copykat, CopykatConfig
from pycopykat.validation.r_runner import run_r_copykat, load_r_prediction, load_r_cna
from pycopykat.validation.metrics import compare_predictions, compare_cna

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "validation_report.md"

DATASETS = ["Gao2021_TNBC", "Peng2019_PDAC", "Tirosh2016_Melanoma"]

def main() -> None:
    rows = []
    for name in DATASETS:
        tsv = ROOT / "data" / "3ca" / name / "counts.tsv"
        if not tsv.exists():
            rows.append({"dataset": name, "status": "missing"})
            continue
        mat = pd.read_csv(tsv, sep="\t", index_col=0)
        out = ROOT / "scratch" / name
        out.mkdir(parents=True, exist_ok=True)
        run_r_copykat(mat, out / "r", sam_name="ref", n_cores=4)
        res = copykat(mat, config=CopykatConfig(n_jobs=4, sam_name="py", output_dir=out/"py"))
        p = compare_predictions(load_r_prediction(out/"r", "ref"), res.prediction)
        c = compare_cna(
            load_r_cna(out/"r", "ref").set_index(["chrom", "chrompos", "abspos"]),
            res.cna_mat, method="spearman",
        )
        rows.append({"dataset": name, **p, **{f"cna_{k}": v for k, v in c.items()}})
    df = pd.DataFrame(rows)
    OUT.write_text("# Validation Report\n\n" + df.to_markdown(index=False))
    print(df)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/validation_report.py
git commit -m "scripts: 3CA validation report generator"
```

---

# M9 — Performance Benchmark & Release

## Task 9.1: `pytest-benchmark` suite

**Files:**
- Create: `tests/benchmarks/test_bench_pipeline.py`

- [ ] **Step 1: Write benchmarks**

```python
# tests/benchmarks/test_bench_pipeline.py
import pyreadr
import pandas as pd
import pytest
from pathlib import Path
from pycopykat import copykat, CopykatConfig

REF_RDA = Path("/media/jason/T7/rerbulid/copykat-R/data/exp.rawdata.rda")

@pytest.mark.benchmark(group="e2e")
def test_bench_e2e(benchmark):
    mat = next(iter(pyreadr.read_r(str(REF_RDA)).values()))
    cfg = CopykatConfig(n_jobs=8, sam_name="bench", output_dir=Path("/tmp"))
    benchmark.pedantic(copykat, args=(mat,), kwargs={"config": cfg},
                       iterations=1, rounds=3)
```

- [ ] **Step 2: Run benchmarks + compare to R**

```bash
uv run pytest tests/benchmarks/ --benchmark-only
# Then time R with:
time Rscript -e 'library(copykat); load("/media/jason/T7/rerbulid/copykat-R/data/exp.rawdata.rda"); set.seed(1234); copykat(rawmat=exp.rawdata, sam.name="rbench", n.cores=1)'
time Rscript -e 'library(copykat); load("/media/jason/T7/rerbulid/copykat-R/data/exp.rawdata.rda"); set.seed(1234); copykat(rawmat=exp.rawdata, sam.name="rbench8", n.cores=8)'
```

Target: Python mean < R single-thread mean / 10 **and** < R 8-thread mean / 3.

- [ ] **Step 3: Iterate** — if targets not met, profile with `py-spy` or `snakeviz`

```bash
uv run python -m cProfile -o profile.out -c "from pycopykat import copykat; import pyreadr; mat = next(iter(pyreadr.read_r('/media/jason/T7/rerbulid/copykat-R/data/exp.rawdata.rda').values())); copykat(mat)"
uv run python -c "import pstats; pstats.Stats('profile.out').sort_stats('cumulative').print_stats(30)"
```

Apply the `python-performance-optimization` skill. Focus first on any single function taking > 20 % of wall time.

- [ ] **Step 4: Record results in `docs/benchmark_report.md`**

- [ ] **Step 5: Commit**

```bash
git add tests/benchmarks/ docs/benchmark_report.md
git commit -m "bench: pipeline benchmarks vs R reference"
```

---

## Task 9.2: Tag V0.1.0

- [ ] **Step 1: Update version**

Edit `pyproject.toml`: `version = "0.1.0"`.

Edit `src/pycopykat/__init__.py`: `__version__ = "0.1.0"`.

- [ ] **Step 2: Write release notes**

Create `RELEASE_NOTES.md`:
```markdown
# pycopykat 0.1.0

First release. Matches R copykat 1.1.0 on `exp.rawdata.rda` at ARI ≥ 0.90
and per-arm Spearman r ≥ 0.95 on diploid/aneuploid prediction.

Runtime on 10k-cell sample: _Fill this field at release time from
`docs/benchmark_report.md` (Task 9.1 output)._

Known limitations:
- hg20 only (mm10 in V2)
- heatmap visual style does not match R heatmap.3 exactly
- subclone detection uses an in-tree dynamicTreeCut port; may diverge on edge cases
```

- [ ] **Step 3: Tag**

```bash
git add pyproject.toml src/pycopykat/__init__.py RELEASE_NOTES.md
git commit -m "release: v0.1.0"
git tag -a v0.1.0 -m "pycopykat 0.1.0"
```

---

# Appendix: Dependency Graph

```
kernels/  ← no internal deps
  ├── distances.py
  ├── kalman.py
  ├── mcmc_pg.py
  └── adjust.py

io/annotation.py    ← data/*.parquet
config.py           ← no deps
result.py           ← pandas only

preprocess/
  ├── filter.py     ← pandas, numpy
  ├── normalize.py  ← numpy
  └── smooth.py     ← kernels/kalman

baseline/
  ├── _shared.py    ← kernels/distances, scipy.cluster
  ├── auto.py       ← _shared, sklearn
  ├── gmm.py        ← auto (type)
  └── synthetic.py  ← _shared

segment/
  ├── breakpoint.py ← kernels/mcmc_pg, scipy.stats
  └── mcmc.py       ← breakpoint, kernels/mcmc_pg, joblib

cna/bins.py         ← pandas, numpy

classify/
  ├── adjust_pipeline.py ← kernels/adjust
  ├── predict.py          ← kernels/distances
  └── subclone.py         ← numpy

viz/heatmap.py      ← matplotlib

pipeline.py         ← imports everything above
cli.py              ← pipeline
validation/         ← r_runner + metrics
```

End of plan.
