#!/usr/bin/env Rscript
# Convert copykat sysdata.rda -> parquet for pycopykat
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

# Cell-cycle genes (cyclegenes is a data.frame with 1 factor column)
stopifnot(exists("cyclegenes"))
writeLines(as.character(cyclegenes[[1]]),
           "data/hg20_cycle_genes.txt")

cat("Wrote:\n",
    "  data/hg20_gene_anno.parquet (", nrow(full.anno), " rows)\n",
    "  data/hg20_220kb_bins.parquet (", nrow(DNA.hg20), " rows)\n",
    "  data/hg20_cycle_genes.txt (", length(cyclegenes[[1]]), " genes)\n",
    sep = "")
