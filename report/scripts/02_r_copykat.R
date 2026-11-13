# Run R copykat 1.2.5 on one task and save its results and intermediates.
# Usage: Rscript 02_r_copykat.R <task dir> <out dir> [n.cores] [mode]
# mode: auto (default), known (the published normal cells as norm.cell.names), cellline, pearson
# Intermediates are captured with trace() on copykat's own functions, so copykat runs unchanged.
suppressPackageStartupMessages({library(copykat); library(Matrix)})
a <- commandArgs(TRUE); task <- a[1]; out <- a[2]; nc <- if (length(a) > 2) as.integer(a[3]) else 1L
mode <- if (length(a) > 3) a[4] else "auto"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
m <- as.matrix(readMM(file.path(task, "counts.mtx")))
rownames(m) <- readLines(file.path(task, "genes.txt")); colnames(m) <- readLines(file.path(task, "cells.txt"))
wbin <- function(x, f) writeBin(as.vector(as.double(x)), file.path(out, f))
ns <- asNamespace("copykat"); dbg <- new.env()
keep <- function(name) bquote(assign(.(name), returnValue(), envir = .(dbg)))
suppressMessages({
  trace("baseline.norm.cl", exit = keep("bnc"), where = ns, print = FALSE)
  trace("baseline.GMM", exit = keep("gmm"), where = ns, print = FALSE)
  trace("CNA.MCMC", exit = keep("mcmc"), where = ns, print = FALSE)
  trace("convert.all.bins.hg20", exit = keep("bins"), where = ns, print = FALSE)
})
pub <- read.table(file.path(task, "published.txt"), sep = "\t", col.names = c("cell", "pred"))
old <- setwd(out)
t0 <- proc.time()[["elapsed"]]
res <- copykat(rawmat = m, id.type = "S", ngene.chr = 5, win.size = 25, KS.cut = 0.1, sam.name = "r",
               distance = if (mode == "pearson") "pearson" else "euclidean",
               norm.cell.names = if (mode == "known") pub$cell[pub$pred == "N"] else "",
               cell.line = if (mode == "cellline") "yes" else "no",
               output.seg = "FALSE", plot.genes = "FALSE", genome = "hg20", n.cores = nc)
el <- proc.time()[["elapsed"]] - t0
setwd(old)
if (!is.null(res$prediction)) write.table(res$prediction, file.path(out, "prediction.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
writeBin(as.vector(res$hclustering$merge), file.path(out, "hc_merge.bin"), size = 4)
cna <- data.matrix(res$CNAmat[, -(1:3)])
writeLines(colnames(cna), file.path(out, "cna_cells.txt")); wbin(cna, "cna.bin")
b <- dbg$bnc
if (!is.null(b)) {
  writeLines(names(b$cl), file.path(out, "smooth_cells.txt")); writeLines(as.character(b$cl), file.path(out, "bnc_cl.txt"))
  writeLines(b$preN, file.path(out, "bnc_preN.txt")); wbin(b$basel, "bnc_basel.bin")
  writeLines(c(b$WNS, if (!is.null(dbg$gmm)) c("GMM", dbg$gmm$WNS, dbg$gmm$preN) else character()), file.path(out, "wns.txt"))
}
writeLines(as.character(dbg$mcmc$breaks), file.path(out, "breaks.txt")); wbin(dbg$mcmc$logCNA, "logcna.bin")
wbin(data.matrix(dbg$bins$RNA.adj[, -(1:3)]), "bins.bin")
writeLines(c(sprintf("elapsed_s\t%.3f", el), sprintf("n_cores\t%d", nc), sprintf("logcna_dim\t%d\t%d", nrow(dbg$mcmc$logCNA), ncol(dbg$mcmc$logCNA)),
             sprintf("cna_dim\t%d\t%d", nrow(cna), ncol(cna))), file.path(out, "runinfo.tsv"))
cat("done", el, "s\n")
