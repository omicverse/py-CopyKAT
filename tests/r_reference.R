#!/usr/bin/env Rscript
# Run upstream R copykat on the canonical exp.rawdata.rda fixture and
# emit results to TSV so tests/test_r_parity.py can diff them offline.
#
# Usage:
#   Rscript tests/r_reference.R
#
# Output:
#   tests/r_out/exp_rawdata_prediction.tsv — per-cell cell.names + copykat.pred
#   tests/r_out/exp_rawdata_meta.tsv       — n_genes, n_cells, n_pred rows

suppressPackageStartupMessages({
  library(copykat)
})

# Discover the canonical fixture. copykat ships exp.rawdata as
# data(exp.rawdata), which is equivalent to loading
# <copykat-installed-path>/data/exp.rawdata.rda.
data(exp.rawdata, package = "copykat")
raw <- exp.rawdata

# Find tests/ relative to this script so the output lands in tests/r_out/
# regardless of where the user invokes Rscript from.
script_args <- commandArgs(trailingOnly = FALSE)
file_arg    <- grep("^--file=", script_args, value = TRUE)
if (length(file_arg) == 1L) {
  script_path <- sub("^--file=", "", file_arg)
  here <- normalizePath(dirname(script_path), mustWork = TRUE)
} else {
  # Fall back: assume CWD is the repo root.
  here <- "tests"
}
out_dir <- file.path(here, "r_out")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("[R] fixture: %d genes x %d cells\n", nrow(raw), ncol(raw)))

# Run copykat with the same deterministic defaults pycopykat uses by
# default (see scripts/run_r_copykat.R).
res <- copykat(
  rawmat        = as.matrix(raw),
  id.type       = "S",
  cell.line     = "no",
  ngene.chr     = 5,
  LOW.DR        = 0.05,
  UP.DR         = 0.1,
  win.size      = 25,
  KS.cut        = 0.1,
  sam.name      = "exp_rawdata",
  distance      = "euclidean",
  output.seg    = "FALSE",
  plot.genes    = "FALSE",
  genome        = "hg20",
  n.cores       = 1L
)

pred <- res$prediction
pred_df <- data.frame(
  cell.names   = as.character(pred$cell.names),
  copykat.pred = as.character(pred$copykat.pred),
  stringsAsFactors = FALSE
)

pred_path <- file.path(out_dir, "exp_rawdata_prediction.tsv")
write.table(
  pred_df,
  file      = pred_path,
  sep       = "\t",
  quote     = FALSE,
  row.names = FALSE
)

meta_path <- file.path(out_dir, "exp_rawdata_meta.tsv")
write.table(
  data.frame(
    key   = c("n_genes_in", "n_cells_in", "n_pred_rows"),
    value = c(nrow(raw), ncol(raw), nrow(pred_df))
  ),
  file      = meta_path,
  sep       = "\t",
  quote     = FALSE,
  row.names = FALSE
)

cat(sprintf("[R] wrote: %s (%d rows)\n", pred_path, nrow(pred_df)))
cat(sprintf("[R] wrote: %s\n", meta_path))
