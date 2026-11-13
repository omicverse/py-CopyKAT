#!/usr/bin/env Rscript
# R copykat 1.2.5 on its example data (exp.rawdata, 302 cells of TNBC1), for tests/test_parity.py.
# Run from tests/:  Rscript r_reference.R
# Writes the input (converted to data/exp_rawdata.npz by tests/make_exp_npz.py) and R's results:
#   data/exp_counts.mtx (Matrix Market), data/exp_genes.txt, data/exp_cells.txt  (the input)
#   data/r_exp_prediction.tsv, data/r_exp_breaks.txt, data/r_exp_preN.txt,
#   data/r_exp_cna_colsums.txt, data/r_exp_cna_rowsums.txt
suppressPackageStartupMessages({library(copykat); library(Matrix)})
data(exp.rawdata, package = "copykat")
m <- as.matrix(exp.rawdata)
dir.create("data", showWarnings = FALSE)
writeMM(Matrix(m, sparse = TRUE), "data/exp_counts.mtx")
writeLines(rownames(m), "data/exp_genes.txt"); writeLines(colnames(m), "data/exp_cells.txt")
ns <- asNamespace("copykat"); dbg <- new.env()
suppressMessages({
  trace("baseline.norm.cl", exit = bquote(assign("bnc", returnValue(), envir = .(dbg))), where = ns, print = FALSE)
  trace("CNA.MCMC", exit = bquote(assign("mcmc", returnValue(), envir = .(dbg))), where = ns, print = FALSE)
})
wd <- getwd(); tmp <- tempfile(); dir.create(tmp); setwd(tmp)
res <- copykat(rawmat = m, id.type = "S", ngene.chr = 5, win.size = 25, KS.cut = 0.1, sam.name = "exp",
               distance = "euclidean", norm.cell.names = "", output.seg = "FALSE", plot.genes = "FALSE",
               genome = "hg20", n.cores = 1)
setwd(wd)
write.table(res$prediction, "data/r_exp_prediction.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
writeLines(as.character(dbg$mcmc$breaks), "data/r_exp_breaks.txt")
writeLines(dbg$bnc$preN, "data/r_exp_preN.txt")
cna <- data.matrix(res$CNAmat[, -(1:3)])
writeLines(sprintf("%.17g", colSums(cna)), "data/r_exp_cna_colsums.txt")
writeLines(sprintf("%.17g", rowSums(cna)), "data/r_exp_cna_rowsums.txt")
