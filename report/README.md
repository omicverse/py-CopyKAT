# Benchmark: pycopykat vs R copykat 1.2.5

The six breast tumours of the CopyKAT paper (GSE148673: DCIS1, TNBC1–5; 532–3,225 cells, 10,424 in
total), with the tumour/normal call that the paper's CopyKAT gave each cell.

```
python  scripts/01_prepare.py <dir with the GSM*_combined_UMIcount_CellTypes_*.txt.gz files> <tasks_dir>
python  scripts/run_all.py <tasks_dir> <bench_dir> <Rscript> <python> 12 1,2      # R, pycopykat 1 and 8 threads
python  scripts/05_figures.py <bench_dir> <tasks_dir> figures
```

Both run copykat with its defaults (`id.type = "S"`, `ngene.chr = 5`, `win.size = 25`, `KS.cut = 0.1`,
`distance = "euclidean"`, `genome = "hg20"`): R calls `copykat(..., n.cores = 1)`, which sets
`set.seed(1234)`; pycopykat runs `pycopykat.core.run` with `seed = 1234`. `02_r_copykat.R` also takes a
mode (`known`, `cellline`, `pearson`) for the other baseline modes and distances. R: copykat 1.2.5
(GitHub) with dlm, MCMCpack, mixtools, parallelDist and transport on R 4.5 (conda-forge). Times are the
faster of two runs, each pinned to idle cores (one for R and for pycopykat with one thread, eight for
pycopykat with eight threads). `figures/comparison_per_dataset.tsv` holds the per-tumour numbers and
`figures/numbers.tex` the numbers quoted in the report.
