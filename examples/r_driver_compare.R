#!/usr/bin/env Rscript
# Run R copykat on a counts TSV and emit stable filenames the comparison
# notebook expects: prediction.tsv (cell, copykat.pred) and cna.tsv
# (chrom, chrompos, abspos, then one column per cell).
#
# Usage:
#   Rscript examples/r_driver_compare.R <counts.tsv> <outdir> [sam_name] [n_cores]
#
# counts.tsv: gene × cell tab-separated table, gene symbols in row.names.
# outdir:     directory to write outputs into. Created if missing.
# sam_name:   optional copykat sam.name (default "compare").
# n_cores:    optional integer (default 1).
#
# Output files (always these names, irrespective of sam_name):
#   prediction.tsv   tsv with columns: cell, copykat.pred
#   cna.tsv          tsv with bin index columns + one column per cell
#   runinfo.txt      key=value: elapsed_min, n_cores, sam_name

suppressPackageStartupMessages({
  library(copykat)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript r_driver_compare.R <counts.tsv> <outdir> [sam_name] [n_cores]")
}

counts_tsv <- args[[1]]
outdir     <- args[[2]]
sam_name   <- if (length(args) >= 3) args[[3]] else "compare"
n_cores    <- if (length(args) >= 4) as.integer(args[[4]]) else 1L

if (!file.exists(counts_tsv)) {
  stop(sprintf("counts file not found: %s", counts_tsv))
}
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("[R] reading counts: %s\n", counts_tsv))
raw <- read.table(
  counts_tsv,
  header = TRUE, sep = "\t", row.names = 1,
  check.names = FALSE, stringsAsFactors = FALSE
)
cat(sprintf("[R] dim: %d genes x %d cells\n", nrow(raw), ncol(raw)))

old_wd <- getwd()
setwd(outdir)
on.exit(setwd(old_wd), add = TRUE)

t0 <- Sys.time()
res <- copykat(
  rawmat     = as.matrix(raw),
  id.type    = "S",
  cell.line  = "no",
  ngene.chr  = 5,
  LOW.DR     = 0.05,
  UP.DR      = 0.1,
  win.size   = 25,
  KS.cut     = 0.1,
  sam.name   = sam_name,
  distance   = "euclidean",
  output.seg = "FALSE",
  plot.genes = "FALSE",
  genome     = "hg20",
  n.cores    = n_cores
)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
cat(sprintf("[R] copykat elapsed: %.2f min\n", elapsed))

pred_df <- data.frame(
  cell = as.character(res$prediction$cell.names),
  copykat.pred = as.character(res$prediction$copykat.pred),
  stringsAsFactors = FALSE
)
write.table(
  pred_df, file = "prediction.tsv",
  sep = "\t", quote = FALSE, row.names = FALSE
)
cat(sprintf("[R] wrote prediction.tsv (%d rows)\n", nrow(pred_df)))

# CNAmat is bins × cells with the first three columns describing the bin
# (chrom, chrompos, abspos) followed by one column per cell.
write.table(
  res$CNAmat, file = "cna.tsv",
  sep = "\t", quote = FALSE, row.names = FALSE
)
cat(sprintf("[R] wrote cna.tsv (%d bins x %d cells)\n",
            nrow(res$CNAmat), ncol(res$CNAmat) - 3L))

writeLines(
  c(
    sprintf("elapsed_min=%.4f", elapsed),
    sprintf("n_cores=%d", n_cores),
    sprintf("sam_name=%s", sam_name)
  ),
  con = "runinfo.txt"
)
cat("[R] done.\n")
