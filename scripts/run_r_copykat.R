#!/usr/bin/env Rscript
# Run R copykat on a single dataset and write results to an output directory.
#
# Usage:
#   Rscript scripts/run_r_copykat.R <counts.tsv> <out_dir> <sam.name> [n.cores]
#
# counts.tsv must be a tab-separated genes × cells table with HGNC symbols as
# row names (first column). Output is copykat's standard set of *_copykat_*
# files written under <out_dir>/ — copykat writes to the current working
# directory, so this script setwd's there first.

suppressPackageStartupMessages({
  library(copykat)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("need <counts.tsv> <out_dir> <sam.name> [n.cores]")
}

counts_tsv <- args[[1]]
out_dir    <- args[[2]]
sam_name   <- args[[3]]
n_cores    <- if (length(args) >= 4) as.integer(args[[4]]) else 1L

if (!file.exists(counts_tsv)) {
  stop(sprintf("counts file not found: %s", counts_tsv))
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("[R] reading counts: %s\n", counts_tsv))
raw <- read.table(
  counts_tsv,
  header = TRUE, sep = "\t", row.names = 1,
  check.names = FALSE, stringsAsFactors = FALSE
)
cat(sprintf("[R] dim: %d genes × %d cells\n", nrow(raw), ncol(raw)))

old_wd <- getwd()
setwd(out_dir)
on.exit(setwd(old_wd), add = TRUE)

t0 <- Sys.time()
res <- copykat(
  rawmat        = as.matrix(raw),
  id.type       = "S",
  cell.line     = "no",
  ngene.chr     = 5,
  LOW.DR        = 0.05,
  UP.DR         = 0.1,
  win.size      = 25,
  KS.cut        = 0.1,
  sam.name      = sam_name,
  distance      = "euclidean",
  output.seg    = "FALSE",
  plot.genes    = "TRUE",
  genome        = "hg20",
  n.cores       = n_cores
)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
cat(sprintf("[R] copykat elapsed: %.2f min\n", elapsed))

# Persist the raw R object too, for easy re-reading
saveRDS(res, file = file.path(".", sprintf("%s_copykat_result.rds", sam_name)))
writeLines(
  sprintf("elapsed_min=%.4f\nn_cores=%d\n", elapsed, n_cores),
  con = sprintf("%s_copykat_runinfo.txt", sam_name)
)
cat("[R] done.\n")
