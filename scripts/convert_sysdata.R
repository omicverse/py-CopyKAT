#!/usr/bin/env Rscript
# Convert copykat sysdata.rda → parquet for pycopykat.
# Inputs: SYSDATA_PATH env var (falls back to /media/jason/T7/rerbulid/copykat-R/data/sysdata.rda).
# Outputs: written to data/ under the pycopykat repo root (resolved via this script's own location).

if (!requireNamespace("arrow", quietly = TRUE)) {
  stop("R package 'arrow' is required. Install with: install.packages('arrow')")
}
suppressPackageStartupMessages(library(arrow))

# Resolve project root as the parent of the scripts/ directory containing this file.
# sys.frame(1)$ofile errors at top-level Rscript (only populated under source()), so
# wrap in tryCatch and fall back to parsing --file= from commandArgs.
get_script_path <- function() {
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(normalizePath(ofile))
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[1])))
  }
  NULL
}

this_file <- get_script_path()
if (is.null(this_file)) {
  # Fallback: assume cwd is the repo root
  project_root <- getwd()
} else {
  project_root <- normalizePath(file.path(dirname(this_file), ".."))
}

sysdata_path <- Sys.getenv("SYSDATA_PATH",
                           unset = "/media/jason/T7/rerbulid/copykat-R/data/sysdata.rda")
if (!file.exists(sysdata_path)) {
  stop("sysdata.rda not found at: ", sysdata_path,
       "\nSet SYSDATA_PATH env var to override.")
}
load(sysdata_path)

data_dir <- file.path(project_root, "data")
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)

stopifnot(exists("full.anno"))
arrow::write_parquet(as.data.frame(full.anno),
                     file.path(data_dir, "hg20_gene_anno.parquet"))

stopifnot(exists("DNA.hg20"))
arrow::write_parquet(as.data.frame(DNA.hg20),
                     file.path(data_dir, "hg20_220kb_bins.parquet"))

stopifnot(exists("cyclegenes"))
writeLines(as.character(cyclegenes[[1]]),
           file.path(data_dir, "hg20_cycle_genes.txt"))

# mm10 mouse gene annotation — only `full.anno.mm10` exists in sysdata.rda;
# R copykat reuses DNA.hg20 bins for mm10 and skips the cycle-gene /
# HLA filter in the mm10 code path. pycopykat mirrors that behaviour;
# we only need the annotation parquet here.
stopifnot(exists("full.anno.mm10"))
arrow::write_parquet(as.data.frame(full.anno.mm10),
                     file.path(data_dir, "mm10_gene_anno.parquet"))

cat("Wrote (to ", data_dir, "):\n",
    "  hg20_gene_anno.parquet (", nrow(full.anno), " rows)\n",
    "  hg20_220kb_bins.parquet (", nrow(DNA.hg20), " rows)\n",
    "  hg20_cycle_genes.txt (", length(cyclegenes[[1]]), " genes)\n",
    "  mm10_gene_anno.parquet (", nrow(full.anno.mm10), " rows)\n",
    sep = "")
