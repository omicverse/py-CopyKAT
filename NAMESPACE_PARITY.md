# NAMESPACE parity — R copykat → pycopykat

Audit of the new-port checklist item:
> NAMESPACE 每个 R export 在 `__all__` 里有 Python 等价物。

Upstream R `copykat/NAMESPACE` exports 8 names. Cross-reference below.

## Table

| # | R export | Python equivalent | Path | In top-level `__all__`? |
|---|---|---|---|---|
| 1 | `copykat` | `copykat` | `pycopykat.pipeline:325` | ✅ yes |
| 2 | `CNA.MCMC` | `segment_cells` | `pycopykat.segment.mcmc:69` | ❌ no (submodule only) |
| 3 | `annotateGenes.hg20` | `annotate_genes`, `annotate_gene_names`, `load_hg20_annotation` | `pycopykat.io.annotation` | ❌ no (submodule only) |
| 4 | `annotateGenes.mm10` | — **missing** | — | ❌ feature gap |
| 5 | `baseline.GMM` | `baseline_gmm` | `pycopykat.baseline.gmm:26` | ❌ no (submodule only) |
| 6 | `baseline.norm.cl` | `baseline_norm_cl` | `pycopykat.baseline.auto:37` | ❌ no (submodule only) |
| 7 | `convert.all.bins.hg20` | `aggregate_to_bins` | `pycopykat.cna.bins:21` | ❌ no (submodule only) |
| 8 | `heatmap.3` | `plot_cna_heatmap` | `pycopykat.viz.heatmap:21` | ❌ no (submodule only) |

## Findings

- **1 genuine feature gap: mm10 / mouse support.** R copykat ships
  `annotateGenes.mm10`; pycopykat only loads `hg20_gene_anno.parquet`
  under `pycopykat.io.annotation`. No mouse genome annotation data or
  entry point exists. This is a port coverage gap, not just an export
  omission.
- **6 functional parity, export layer only.** Items 2, 3, 5, 6, 7, 8 all
  have a Python implementation reachable from a submodule, but none are
  promoted into the top-level `pycopykat.__all__`. The checklist's
  intent ("Python 等价物 ... 在 `__all__`") is about API discoverability
  for `pycopykat` users who don't want to memorise submodule paths;
  these functions are therefore borderline-compliant.
- **1 clean match.** `copykat` is properly the single top-level entry.

## Current `pycopykat.__all__`

```python
__all__ = ["CopykatConfig", "CopykatResult", "copykat", "copykat_by_batch"]
```

(Plus `copykat_by_batch`, a pycopykat-original AnnData batch driver
with no upstream R equivalent — intentionally exported.)

## Recommendations (not yet applied)

Two candidate interventions, pick one:

1. **Promote all 6 to top-level** — mirrors R NAMESPACE 1:1 and satisfies
   the checklist verbatim. Downside: flattens the API, hides the
   `pycopykat.segment / baseline / cna / viz / io` structure that
   currently organises the codebase.
2. **Keep submodule layout, add doc cross-reference** — leave `__all__`
   lean; add an "R → Python symbol map" table to `README.md` or a
   `docs/porting.md` so R users can `Ctrl-F` their way in. This
   preserves the organised layout while making discovery explicit.

The mm10 gap is independent of either choice and should be tracked as
its own work item (`pycopykat.io.annotation.load_mm10_annotation` +
packaged `hg20_gene_anno.parquet` analogue) if mouse support is in
scope.

## Extras (pycopykat-only public surface)

Surface area that goes beyond R NAMESPACE — these are intentional
Python-side additions and do **not** need an R cross-reference:

- `pycopykat.copykat_by_batch` (AnnData batch driver)
- `pycopykat.CopykatConfig`, `pycopykat.CopykatResult`
- `pycopykat.baseline.baseline_synthetic` (synthetic baseline)
- `pycopykat.classify.predict_ploidy`, `classify.dynamic_tree_cut`
- `pycopykat.segment.find_breakpoints`, `find_breakpoints_analytic`
- `pycopykat.validation.{compare_predictions, compare_cna, run_r_copykat}`
- `pycopykat.kernels.{pdist_euclidean, pdist_pearson, pdist_spearman, kalman_smooth, ...}`
- `pycopykat.cli.app` (typer CLI)

These reflect pycopykat's testing / validation / performance concerns
and don't map to R NAMESPACE by design.
