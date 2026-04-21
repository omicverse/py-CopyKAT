# Handoff — py↔R comparison notebooks (2026-04-21)

This document records what was added in the comparison-notebook session and
where to pick up next time. It complements the README's user-facing
"py↔R comparison notebooks" section.

## What this session delivered

Two commits on `main`:

1. **`980000f docs(parity): document Lee/SMC16 known parity gap + bump to 0.1.0.dev1`**
   - Pure-docs change documenting the SMC16 outlier in the 17-patient sweep.
   - No algorithm or test changes.
2. **`<this commit> feat(examples): py-vs-R comparison notebooks + viz extension`**
   - New notebooks, viz helpers, R driver, builder updates.
   - No changes to `pycopykat.{pipeline, baseline, segment, classify, cna,
     preprocess, kernels}.py`.

## File map

| Path | Type | Purpose |
|---|---|---|
| `examples/compare_py_vs_R.ipynb` | new notebook | exp.rawdata side-by-side; runs both implementations inline |
| `examples/compare_py_vs_R_realdata.ipynb` | new notebook | three-sample sweep (Gao/TNBC1, Kim/P0019, Qian/11) reading cached `*_out/` files; summary table + wall-clock bar chart + per-sample omicverse UMAP overlays + heatmap_compare/delta. SMC16 explicitly excluded — its parity gap is documented in `FINDINGS.md` instead |
| `examples/r_driver_compare.R` | new script | thin wrapper around `copykat()` that writes stable filenames (`prediction.tsv`, `cna.tsv`, `runinfo.txt`) |
| `examples/_build_notebooks.py` | rewritten | adds `compare` and `compare-realdata` targets, defaults to `omicverse` Jupyter kernel |
| `pycopykat/viz/heatmap.py` | extended | adds `plot_cna_heatmap_compare`, `plot_cna_delta`; existing `plot_cna_heatmap` keeps signature; removed top-level `matplotlib.use("Agg")` so notebooks can render inline |
| `pycopykat/viz/__init__.py` | updated | exports the two new helpers |
| `tests/unit/test_viz_heatmap.py` | extended | covers the two new helpers |
| `pyproject.toml` | updated | new `[project.optional-dependencies].examples` extras (`omicverse`, `scanpy`, `matplotlib_venn`, `ipykernel`, `nbformat`, `nbclient`) |
| `NAMESPACE_PARITY.md` | extended | adds the two new viz helpers to the "Extras" section |
| `README.md` | extended | new "py↔R comparison notebooks" section above the 17-patient benchmark section |
| `.gitignore` | extended | ignores `examples/_compare_out_*/` working dir |
| `benchmarks/full/FINDINGS.md` | extended (prior commit) | "Known parity gap: Lee2020_Colorectal / SMC16" subsection |
| `benchmarks/full/TODO_parity_SMC16.md` | new (prior commit) | three-step root-cause investigation plan |

## How to regenerate the executed notebooks

The notebooks need a Python env with `omicverse + scanpy + matplotlib_venn + pyreadr + pycopykat + ipykernel`. On Jason's machine the conda env
`omicverse` already has all of these (after this session's `pip install
pyreadr matplotlib_venn` and `pip install -e .` for pycopykat). The Jupyter
kernel `omicverse` is pre-registered and points at that env.

```bash
cd /media/jason/T7/rerbulid/pycopykat

# build + execute both compare notebooks in place
/home/jason/miniforge3/envs/omicverse/bin/python examples/_build_notebooks.py compare compare-realdata

# or one at a time
/home/jason/miniforge3/envs/omicverse/bin/python examples/_build_notebooks.py compare
/home/jason/miniforge3/envs/omicverse/bin/python examples/_build_notebooks.py compare-realdata

# build .ipynb without running
/home/jason/miniforge3/envs/omicverse/bin/python examples/_build_notebooks.py --no-execute
```

`PYCOPYKAT_KERNEL=<other-kernel>` overrides the default `omicverse` kernel
(useful when running on a fresh machine that registered a different name).

The exp.rawdata notebook calls `Rscript` via subprocess — set
`RSCRIPT=/path/to/Rscript` if it isn't on `$PATH`.

## Notebook-specific gotchas

### `compare_py_vs_R.ipynb` (exp.rawdata)

- Working dir: `examples/_compare_out_exp_rawdata/` (gitignored). Holds
  `counts.tsv` + `r_out/{prediction.tsv, cna.tsv, runinfo.txt}`.
- R copykat takes ~1–3 min on this fixture with `n_cores=1`. The R run
  is cached: deleting `r_out/prediction.tsv` forces a rerun.
- Expected results: per-cell CNA Pearson median ≈ 0.97; classification
  ARI = 1.000.
- Cell-ID normalisation: R's `make.names()` rewrites `-` to `.` in cell
  IDs. The notebook reverses this in §4.

### `compare_py_vs_R_realdata.ipynb` (3-sample sweep)

- Samples: Gao2021_Breast/TNBC1 (1,097 cells), Kim2020_Lung/P0019
  (2,945), Qian2020_Ovarian/11 (6,972) — chosen for one-cancer-each
  + cell-count ladder + speedup envelope (23× → 47× → 82×).
- Reads cached files under
  `benchmarks/full/<cancer>/<sample>/{r_out,py_out}/`. **These paths
  are gitignored** (`/benchmarks/full/*/`), so on a fresh clone the
  cached outputs are absent and the notebook will fail to re-execute.
  The committed `.ipynb` carries the executed outputs so GitHub still
  renders correctly. To regenerate the underlying caches:
  ```bash
  uv run python scripts/run_all_benchmarks.py    # ~5–6 hours for the full sweep
  ```
- Wall-clock numbers come from the cached `*_runinfo.txt` of each
  pipeline (key `elapsed_min`). The bar chart in §2 reads these
  directly, so re-running the underlying benchmark refreshes the
  comparison automatically.
- Expected results per sample: ARI 0.994 – 1.000, per-cell CNA Pearson
  median 0.85 – 0.97, py wall-clock 0.10 – 0.44 min, R wall-clock 2.37
  – 35.90 min, speedup 23.5× / 46.8× / 82.4×.
- The omicverse pipeline (preprocess → PCA → Leiden → UMAP) runs
  per-sample on actual counts.tsv data — the 6,972-cell ovarian sample
  is the slowest panel, ~3–5 min on CPU.
- Switching sample set: edit `REALDATA_SAMPLES` (a list of
  `(cancer, sample)` tuples) in `examples/_build_notebooks.py`. Any
  subdirectory under `benchmarks/full/<cancer>/<sample>/` with the
  standard `r_out/` + `py_out/` layout works. SMC16 was explicitly
  excluded — adding it back would mix the demo case with the
  documented parity outlier.

## Open items (not yet done)

These are explicit non-deliverables of this session:

1. **SMC16 root-cause investigation.** See
   `benchmarks/full/TODO_parity_SMC16.md`. Three-step probe (preN
   identity, basel identity, predict_ploidy with R inputs) needed
   before any patch to `baseline_gmm` / `predict_ploidy`. Patching blind
   risks regressing the 16 other patients (current ARI ≥ 0.92).
2. **Full 17-patient gallery.** The realdata notebook covers 3 of 17
   samples (one per cancer for Breast / Lung / Ovarian, all high-ARI).
   A loop-style "every patient on one page" notebook would reuse the
   same plotting helpers but is not in scope here.
3. **Rerun the 17-patient sweep with `0.1.0.dev1`.** No algorithm
   changes since the last sweep, so numbers carry forward unchanged —
   but if the SMC16 probe lands a fix, a re-sweep + `FINDINGS.md`
   refresh + `overview.png` regen is required.

## Decisions taken (rationale captured here so future-Jason or another
collaborator doesn't need to re-derive)

- **Two notebooks, not one with a switch.** `exp.rawdata` (run inline)
  and the realdata sweep (read cached) have very different I/O paths
  and runtimes; a single switch-driven notebook would have hidden the
  contract.
- **Realdata notebook ships 3 high-ARI samples, not SMC16.** Initial
  scope used SMC16 as a single-sample demo (matching the FINDINGS.md
  parity-gap discussion). On Jason's instruction the demo notebook
  was repointed to high-ARI matched samples (one per cancer) so it
  can stand on its own as a "look how well it tracks R, and how much
  faster" story. SMC16 stays documented in FINDINGS.md and is
  intentionally excluded from `REALDATA_SAMPLES` to avoid mixing the
  positive demo with the open parity outlier. Switch sample set by
  editing `REALDATA_SAMPLES` in `_build_notebooks.py`.
- **CNA heatmap module extended, not rewritten.** The existing
  `plot_cna_heatmap` signature (`output=` PNG path) is preserved so
  benchmark scripts aren't disrupted. Two new helpers
  (`plot_cna_heatmap_compare`, `plot_cna_delta`) accept `ax=` for inline
  notebook use. Top-level `matplotlib.use("Agg")` was removed (it was
  user-hostile in interactive contexts) — callers that need it (e.g.
  benchmark scripts) can call it themselves.
- **R driver `examples/r_driver_compare.R` is separate from
  `scripts/run_r_copykat.R`.** The compare driver writes
  notebook-friendly stable filenames; the benchmark runner writes
  copykat's full diagnostic file set under `<sam_name>_copykat_*` names
  for downstream aggregation. Both kept, no consolidation.
- **`tests/r_reference.R` is not modified.** Compared to extending it,
  a new compare driver costs less risk: `tests/r_reference.R` is the
  contract for `tests/test_r_parity.py` and any change there could
  invalidate the existing parity tests.

## Quick verification

After this session the following all return clean:

```bash
uv run pytest tests/unit/test_viz_heatmap.py -xvs       # 4 passed
uv run pytest tests/unit -x                              # full unit suite
uv run python examples/_build_notebooks.py --no-execute  # builds 3 .ipynb
```

Both committed compare notebooks have 14/14 (exp.rawdata) and 21/22
(realdata; the helper-defining cell intentionally produces no output)
code cells executed without errors.

## Pointers

- Parity audit: `NAMESPACE_PARITY.md`
- 17-patient findings: `benchmarks/full/FINDINGS.md`
- Open algorithm investigation: `benchmarks/full/TODO_parity_SMC16.md`
- R upstream source for the disputed branch:
  `copykat-R/R/baseline.norm.cl.R`,
  `copykat-R/R/baseline.GMM.R`, `copykat-R/R/copykat.R:165-176, 419-422`
- pycopykat counterparts:
  `pycopykat/baseline/auto.py`, `pycopykat/baseline/gmm.py`,
  `pycopykat/pipeline.py:157-170, 282-287`
