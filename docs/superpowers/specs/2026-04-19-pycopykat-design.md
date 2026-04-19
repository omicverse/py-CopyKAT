# pycopykat — CopyKAT Python 重构设计文档

- **作者**: Jason（主导）+ Claude（协作）
- **日期**: 2026-04-19
- **状态**: Draft — 待用户审阅
- **源仓库**: `/media/jason/T7/rerbulid/copykat-R`（R 版 1.1.0, Ruli Gao）
- **目标仓库**: `/media/jason/T7/rerbulid/pycopykat`

---

## 1. Motivation

R 版 CopyKAT（Gao et al. *Nat Biotechnol* 2021）是单细胞 RNA-seq 数据上推断基因组拷贝数变异（CNA）并区分肿瘤（aneuploid）与正常（diploid）细胞的标准工具。其主要瓶颈：

- **速度慢**：Poisson-Gamma MCMC + 成对距离 + 每细胞 Kalman 平滑，R 原生循环在 ≥ 5k 细胞时单样本常需数小时
- **依赖 R 生态**：`mixtools`、`MCMCpack`、`parallelDist`、`dlm` 等包在 Python-first 工作流（scanpy/anndata）中集成不便

**目标**：用 Python + Numba（CPU 主栈，CuPy GPU 可选）重构，保持**统计等价**的前提下，**在典型数据集（~10k 细胞）上获得 ≥ 10× 加速**，并提供 CLI + Library 双接口、原生 AnnData 导出。

---

## 2. Scope (V1)

### In Scope

| 编号 | 功能 | R 版对应 |
|---|---|---|
| F1 | hg20 基因注释 | `annotateGenes.hg20` |
| F2 | 基因 → 220KB bin 聚合 | `convert.all.bins.hg20` |
| F3 | 三种 baseline 模式（auto / known-normal / cell.line） | `baseline.norm.cl`, `baseline.GMM`, `baseline.synthetic` |
| F4 | Kalman 平滑 | `dlm::dlmSmooth` |
| F5 | Poisson-Gamma MCMC 分段 + KS 断点 | `CNA.MCMC` |
| F6 | Aneuploid / Diploid 二分类 | copykat.R lines 345-399 |
| F7 | 亚克隆层次聚类 | `hclust` + Ward |
| F8 | R-兼容 `.txt` 输出 + 可选 `.h5ad` | copykat.R lines 444-530 |
| F9 | matplotlib 简版 heatmap（PNG/PDF） | `heatmap.3`（简化实现） |
| F10 | CLI（`pycopykat run ...`）+ Library API（`pycopykat.copykat(...)`） | — |

### Out of Scope (V2+)

- mm10 小鼠基因组（仅预留 hook）
- `heatmap.3` 像素级还原
- 分布式（dask/ray）后端
- CuPy GPU 后端（V1 只留接口）

---

## 3. Acceptance Criteria (保真度 B 档)

### 内置回归（严格）

在 `copykat-R/data/exp.rawdata.rda` 上对比 R 版与 Python 版：

| 指标 | 阈值 |
|---|---|
| aneuploid/diploid 标签 ARI | ≥ 0.90 |
| aneuploid/diploid 标签 Cohen's κ | ≥ 0.90 |
| 按染色体 arm 聚合的 CNA score — Spearman r（逐细胞中位数）| ≥ 0.95 |
| 亚克隆数差异 | ≤ 1 |
| 亚克隆 Fowlkes-Mallows index | ≥ 0.85 |

R 参考输出：已存在 `copykat-R/test_output/`。

### 3CA 外部验证（简单）

三个数据集：TNBC（Gao 2021）、PDAC（Peng 2019）、Melanoma（Tirosh 2016）。
**验证标准**（简化，不如内置回归严格）：
- 自动化脚本通过 `subprocess` 调 R 生成 reference，再调 Python 版生成待测输出
- 主指标：aneuploid/diploid 标签 ARI ≥ 0.80（阈值较内置 0.90 放宽，因外部数据变数更多）
- 辅助指标：染色体 arm CNA Spearman r ≥ 0.90
- **通过条件：3 个数据集中至少 2 个达标**；未达标的数据集必须有根因分析报告（例如测序平台差异、细胞数极少、baseline 自动检测失败等）

### 性能目标

- 在 ~10k 细胞数据（TNBC）上 ≥ 10× 加速 vs R 单线程
- 在 ~10k 细胞数据（TNBC）上 ≥ 3× 加速 vs R 8 线程（`n.cores=8`）

### 可重现性

Python 版使用固定 `numpy.random.default_rng(seed=1234)` 和 `numpy.random.default_rng(seed=123)`（对照 R 的两处 `set.seed`）。
**不保证 bit-exact**，但同一输入 + 同一 seed → 同一输出（within Python 侧）。

---

## 4. Architecture — 分层 Pipeline（方案 B）

```
pycopykat/
├── pyproject.toml                 # uv + hatch build
├── README.md
├── docs/superpowers/specs/        # design doc（本文件所在）
├── docs/superpowers/plans/        # 实施 plan
├── src/pycopykat/
│   ├── __init__.py                # 顶层 copykat() API
│   ├── cli.py                     # typer/click CLI
│   ├── pipeline.py                # 主流程编排（对应 R copykat.R 主函数）
│   ├── config.py                  # dataclass CopykatConfig（所有超参）
│   ├── io/
│   │   ├── annotation.py          # 基因注释 hg20 加载
│   │   └── reference.py           # sysdata.rda → parquet 转换脚本
│   ├── preprocess/
│   │   ├── filter.py              # F1: stage 1 细胞/基因过滤
│   │   ├── normalize.py           # F4 前半: VST + 中心化
│   │   └── smooth.py              # F4 后半: Kalman 平滑
│   ├── baseline/
│   │   ├── auto.py                # baseline.norm.cl 等价
│   │   ├── gmm.py                 # baseline.GMM 等价
│   │   └── synthetic.py           # baseline.synthetic 等价
│   ├── segment/
│   │   ├── mcmc.py                # F5: Poisson-Gamma MCMC sampler
│   │   └── breakpoint.py          # F5: KS-based 分段
│   ├── cna/
│   │   └── bins.py                # F2: 基因 → 220KB bin 聚合
│   ├── classify/
│   │   └── predict.py             # F6 + F7: 二聚 + diploid/aneuploid 标签 + subclone
│   ├── viz/
│   │   └── heatmap.py             # F9: matplotlib heatmap
│   ├── kernels/                   # 🔥 Numba 热点集中地
│   │   ├── __init__.py
│   │   ├── distances.py           # pdist Euclidean / Pearson / Spearman
│   │   ├── kalman.py              # 手写 Kalman recursion (Numba)
│   │   ├── mcmc_pg.py             # Poisson-Gamma sampler (Numba)
│   │   └── adjust.py              # per-cell 阈值调整 (Numba)
│   │   # 注：V2 增加 GPU 后端时再引入 _dispatch.py，V1 YAGNI
│   └── validation/
│       ├── r_runner.py            # subprocess 调 R 产生 reference
│       ├── metrics.py             # ARI / κ / Spearman / FMI
│       └── compare.py             # 端到端对比入口
├── data/                          # 参考数据（注释 + 测试数据）
│   ├── hg20_gene_anno.parquet     # 从 sysdata.rda 转出
│   ├── hg20_cycle_genes.txt
│   └── hg20_220kb_bins.parquet
└── tests/
    ├── conftest.py
    ├── test_regression.py         # 内置 exp.rawdata 严格回归
    ├── test_kernels.py            # Numba kernel 单元 + pytest-benchmark
    ├── test_3ca.py                # 3CA 三数据集 smoke test
    └── data/                      # 小型 golden snippets
```

### 数据流

```
raw counts (genes × cells)
  → [filter] → filtered matrix
  → [annotate] → genes with chrom/pos
  → [normalize] → log(sqrt(x) + sqrt(x+1)) centered
  → [smooth]   → per-cell Kalman
  → [baseline]  → select diploid candidates, compute baseline vec
  → [relative]  → subtract baseline
  → [segment]   → Poisson-Gamma MCMC + KS → segmented matrix
  → [bin]       → 220KB genomic bins
  → [classify]  → k=2 Ward → {aneuploid, diploid} labels
  → [subclone]  → hclust on aneuploid cells → subclone labels
  → [output]    → CNA.txt / prediction.txt / heatmap.png / (optional) .h5ad
```

### 接口定义

```python
# Library
from pycopykat import copykat
result = copykat(
    rawmat: pd.DataFrame | np.ndarray | ad.AnnData,  # genes × cells
    id_type: Literal["Symbol", "Ensembl"] = "Symbol",
    genome: Literal["hg20"] = "hg20",           # V1 仅 hg20
    ngene_chr: int = 5,
    min_gene_per_cell: int = 200,
    low_dr: float = 0.05,
    up_dr: float = 0.1,
    win_size: int = 25,
    ks_cut: float = 0.1,
    distance: Literal["euclidean", "pearson", "spearman"] = "euclidean",
    cell_line: bool = False,
    norm_cell_names: list[str] | None = None,
    sam_name: str = "",
    output_dir: str | Path = ".",
    output_seg: bool = False,
    output_h5ad: bool = False,
    n_jobs: int = 1,
    seed: int = 1234,
    backend: Literal["cpu"] = "cpu",              # V1 仅 cpu；V2 扩 "gpu"
) -> CopykatResult
# CopykatResult: dataclass with .cna_mat (pd.DataFrame), .prediction (pd.DataFrame),
#                .hclustering (scipy.cluster.hierarchy.linkage), .subclone (pd.Series)

# CLI
$ pycopykat run --input raw.csv --id-type Symbol --n-jobs 8 --output-dir out/
$ pycopykat validate --reference-r /path/to/r/output --test-py /path/to/py/output
```

---

## 5. Key Algorithm Decisions

### 5.1 Kalman 平滑（`kernels/kalman.py`）

R 版用 `dlm::dlmModPoly(order=1, dV=0.16, dW=0.001)` + `dlmSmooth`。这是一个一阶多项式状态空间模型，等价于 random walk with observation noise。

**Python 实现**：手写 Numba 版 RTS smoother（forward Kalman + backward smoother），内部 2×2 矩阵运算可完全手展开。
- 避免 `pykalman`（纯 Python，慢）与 `statsmodels`（通用但重）
- 每个细胞独立 → `@njit(parallel=True)` 在 cell 维并行
- 固定参数 dV=0.16、dW=0.001（与 R 版对齐）

### 5.2 Poisson-Gamma MCMC（`kernels/mcmc_pg.py`）

**已验证**：直接读 `MCMCpack::MCpoissongamma` 源码（2026-04-19）确认实现是：

```r
rgamma(mc, shape = alpha + sum(y), rate = beta + n)
```

即**闭式共轭 Gamma 采样**，无 Metropolis-Hastings，无 burn-in。1000 次 draw 是 1000 次 i.i.d. Gamma 样本。

**Python 等价**（R 的 `rgamma(rate=)` 对应 `numpy` 的 `scale=1/rate`）：

```python
rng.gamma(shape=alpha + y.sum(), scale=1.0 / (beta + len(y)), size=mc)
```

**加速策略**：
- 单次采样是 O(mc)，但外层循环（断点候选 × 聚类 × 细胞）可达 10^5+ 次调用
- Numba `@njit(parallel=True)` 在外层 cell 维并行
- Numba 内 `np.random.gamma` 可用，RNG 状态通过 `numba.random.seed()` 在每次 kernel 调用前设置

**风险**：已消除（源码验证）。

### 5.3 成对距离（`kernels/distances.py`）

- Euclidean: `scipy.spatial.distance.pdist(X, "euclidean")` 已经是 C 实现，够快
- Pearson/Spearman 距离 (1 - r)：Numba `@njit(parallel=True)` 手写，避免 scipy 逐对循环开销
- 10k × 10k 距离矩阵 ≈ 400 MB float32，内存可控

### 5.4 GMM（`baseline/auto.py`, `baseline/gmm.py`）

- `sklearn.mixture.GaussianMixture(n_components=3, means_init=[-0.2, 0, 0.2], covariance_type='full')`
- 固定 `random_state=seed`
- `max_iter` 需在 M2 开发时读 `mixtools::normalmixEM` 源码核实默认值（通常 1000），然后显式对齐
- 单元测试对比 R 和 Python 在同一 toy 输入上的 μ/σ/π

### 5.5 RNG 策略

- 顶层 `rng = np.random.default_rng(seed=1234)`
- `baseline.synthetic` 用子 stream：`rng_synthetic = np.random.default_rng(seed=123)`
- MCMC 采样传入 Numba kernel 前生成大批 uniform samples（Numba 里的 RNG 有限制）

---

## 6. Validation Architecture

```
validation/
├── r_runner.py         # 调 R：devtools::load_all("/media/jason/T7/rerbulid/copykat-R") + copykat(...)
├── metrics.py          # ARI, Cohen κ, Spearman r (per-arm), FMI
└── compare.py          # 统一入口：r_out_dir + py_out_dir → report
```

**R runner 脚本模板**（`validation/r_runner.py` 生成）：
```r
devtools::load_all("/media/jason/T7/rerbulid/copykat-R")
rawmat <- readRDS(Sys.getenv("PYCOPYKAT_INPUT_RDS"))
set.seed(1234)
res <- copykat(rawmat=rawmat, id.type="S", sam.name="ref", n.cores=8, ...)
saveRDS(res, Sys.getenv("PYCOPYKAT_OUTPUT_RDS"))
```

Python 通过 `subprocess.run(["Rscript", script_path])` 调用，然后用 `pyreadr` 读 rds 结果做对比。

---

## 7. Dev Environment

| 项目 | 选型 |
|---|---|
| Python | `>=3.10,<3.13` |
| 环境 | `uv venv` + `pyproject.toml` |
| 构建后端 | `hatchling` |
| 测试 | `pytest` + `pytest-benchmark` + `hypothesis` |
| 格式化/Lint | `ruff` + `mypy --strict` |
| 核心依赖 | `numpy>=1.26`, `scipy>=1.11`, `pandas>=2.1`, `scikit-learn>=1.4`, `numba>=0.59`, `matplotlib>=3.8`, `anndata>=0.10`, `pyreadr>=0.5`, `typer>=0.9` |
| Dev 依赖 | `pytest`, `pytest-benchmark`, `hypothesis`, `ruff`, `mypy`, `ipykernel` |
| CI | 本地 `pytest`；无 GitHub Actions |
| R 版本 | 4.5.2（系统） |
| R 调用 | `subprocess` + `Rscript`（不走 rpy2） |

---

## 8. Risks & Mitigations

| 风险 | 影响 | 缓解 |
|---|---|---|
| MCMC 实现偏差导致分段结果发散 | 高 | 单元测试对比 R `MCpoissongamma` 分布；保留 MH fallback |
| Numba 编译 overhead 在小数据上反而变慢 | 中 | AOT cache (`cache=True`) + 首次运行 warm-up |
| R 版 `hclust` ties 处理顺序与 `scipy.cluster.hierarchy.linkage` 不同 | 中 | 树高比较 + Fowlkes-Mallows 而非精确匹配 |
| `sklearn.mixture.GaussianMixture` 与 `mixtools::normalmixEM` 在 edge 数据上分量不同 | 中 | 初值严格对齐（μ=[-0.2,0,0.2]）；多次 restart |
| hg20 注释数据从 `sysdata.rda` 转出格式不对 | 低 | 首批 commit 提供 `scripts/convert_sysdata.R` 并校验 |
| 3CA 数据下载慢或需要登录 | 中 | 文档写清获取方式；失败则跳过该项验证 |
| `pyreadr` 无法读取某些复杂 R 对象（如 `copykat` 返回的嵌套 list） | 中 | R runner 脚本内把结果拆解为 `.csv/.tsv` 原子文件后再读，避免依赖 `pyreadr` 解析复杂对象 |
| Numba `@njit` 里 `np.random` 行为与顶层 `default_rng` 不一致 | 中 | RNG 策略统一：MCMC 采样在 Python 层生成 uniform draws 批量传入 Numba kernel |
| `mixtools::normalmixEM` 使用 Newton-Raphson M-step，而 `sklearn.mixture.GaussianMixture` 使用标准 EM | 中 | 这是**已知算法差异**而非实现瑕疵。单元测试对比 μ/σ/π 容忍度放宽到 5%；必要时在 `baseline/` 内写轻量 Newton-Raphson EM 以对齐 |
| `dynamicTreeCut` 移植与 R 版在 edge 数据上 subclone 数量分歧 | 中 | 10.1 节已定义 fallback 机制 |

---

## 9. Milestones

| M | 内容 | 产出 |
|---|---|---|
| M1 | 项目骨架 + `pyproject.toml` + sysdata 转换 | 可 `uv sync`，hg20 parquet 就绪 |
| M2 | `kernels/` 四个 Numba 模块 + 单元测试 | 所有 kernels 过单元测试 |
| M3 | Pipeline 前段：filter + annotate + normalize + smooth | 可跑 `exp.rawdata` 到 smoothed 矩阵 |
| M4 | Baseline 三种模式 + relative expression | 三路都能跑通 |
| M5 | Segmentation (MCMC + KS) + bin conversion | 输出 bin-level CNA 矩阵 |
| M6 | Classify + subclone + heatmap | CLI 可输出 `.txt` + heatmap PNG |
| M7 | Validation harness + 内置回归达标（B 档阈值）| `pytest test_regression.py` 全绿 |
| M8 | 3CA 三数据集 smoke test | 2/3 达标，分析第 3 个根因 |
| M9 | 性能调优 + benchmark 报告 | ≥ 10× vs R 单线程 |

---

## 10. Key Implementation Decisions

### 10.1 Subclone Detection（必须在 spec 层定案）

R 版 copykat 用 `dynamicTreeCut::cutreeDynamic`（hybrid method, Langfelder et al. 2007）对 aneuploid 细胞的 hclust 树做自适应剪切。**这是影响 subclone 数量和 FMI 验收指标的核心步骤**。

**V1 决策：在 `pycopykat/classify/subclone.py` 内移植 dynamicTreeCut hybrid 算法**
- 算法文档完整（Langfelder et al. 2007, Bioinformatics, DOI: 10.1093/bioinformatics/btm563）
- 实现 ≈ 200 LOC，无外部依赖
- 参考 PyPI `dynamictreecut`（kylessmith 作者）作为对照校验，但**不运行时依赖**（该包已多年不更新）
- 单元测试：在 scipy 合成 dendrogram 上对比 R `cutreeDynamic` 与 Python 移植输出（FMI ≥ 0.95）

**Fallback（如移植失败）**：V1 降级为 silhouette-based k-selection（k=2..10 elbow），同时将 Section 3 的验收指标 "亚克隆数差异 ≤ 1 / FMI ≥ 0.85" 放宽为 "亚克隆数差异 ≤ 2 / FMI ≥ 0.70"，并在 RELEASE NOTES 标注此限制。

### 10.2 Sparse & AnnData Input Path

- Python 版接受 `AnnData`，约定 `adata.X` 的 **行是 cells，列是 genes**（scanpy 标准，与 R 版 `genes × cells` 相反）
- 内部立即 `.T` 转置为 `genes × cells` 统一表示，后续 pipeline 全程 `genes × cells`
- 如 `adata.X` 是 sparse (CSR/CSC)：在 filter 阶段（F1）保持 sparse；normalize (VST) 后转 dense（CopyKAT 的下游算法无 sparse 实现，强行 sparse 得不偿失）
- 大数据集（>30k 细胞）dense 内存：22k genes × 30k cells × 4 bytes ≈ 2.6 GB，当前硬件足够

### 10.3 次要 Open Questions（真正非阻塞）

1. heatmap 侧边染色体 color bar 颜色是否必须和 R 版完全一致？（V1 默认：不必，用 matplotlib 默认 palette）
2. `output_seg=True` 时 `.seg` 文件格式是否按 IGV 规范严格输出？（V1 默认：按 IGV 规范）

---

## 11. License & Derivative Work

R 版 CopyKAT 的 `DESCRIPTION` 声明 `License: GPL-2`。本 Python 重构是**通过阅读 R 源码**做出的移植实现，在法律上属于**衍生作品（derivative work）**。

**决策**：
- `pycopykat` 以 **GPL-2.0-or-later** 发布（GPL-2 兼容）
- `LICENSE` 文件使用 GPL-2.0 标准全文
- 顶层 `README.md` 致谢 Ruli Gao 及原 CopyKAT 作者
- 每个从 R 版翻译的函数在 docstring 中注明：`"Port of <R函数名> from copykat (Gao et al. 2021, GPL-2)"`

如 Jason 有商业闭源发布需求，需在 V1 提交前和我澄清——此时要么联系原作者取得重新授权，要么只做"clean-room reimplementation"（不看 R 源码，仅按论文重写），整个开发流程需要重走。
