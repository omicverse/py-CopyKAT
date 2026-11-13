# Reference values from R for tests/test_rcompat.py (run from tests/: Rscript r_components.R).
# Writes tests/data/r_components.json; needs R with MCMCpack-free base R and mixtools.
suppressPackageStartupMessages({library(mixtools); library(jsonlite)})
out <- list()
# rgamma on R's stream (set.seed(1234)), several shapes incl. a < 1 and each GD branch
set.seed(1234)
sh <- c(0.5, 1.2, 3.7, 13.5, 75.3, 400); rt <- c(1, 2, 26, 27, 40, 101)
out$rgamma <- list(shape = sh, rate = rt, draws = lapply(seq_along(sh), function(i) rgamma(200, sh[i], rt[i])), next_unif = runif(3))
set.seed(7)
sh2 <- 5 + 300 * runif(300); rt2 <- 1 + round(40 * runif(300))
set.seed(1234)
out$means <- list(shape = sh2, rate = rt2, mean = sapply(seq_along(sh2), function(i) mean(rgamma(1000, sh2[i], rt2[i]))), next_unif = runif(3))
# rnorm with sd 0 entries (baseline.synthetic)
set.seed(123); out$rnorm <- list(sd = c(0.1, 0, 2, 0, 0.5), x = sapply(c(0.1, 0, 2, 0, 0.5), function(s) rnorm(1, 0, s)))
# hclust / cutree
set.seed(5); x <- matrix(rnorm(120 * 15), 120); x[1:40, ] <- x[1:40, ] + 1.5
d <- dist(x)
out$hclust <- list(x = x)
for (m in c("ward.D", "ward.D2")) { h <- hclust(d, m); out$hclust[[m]] <- list(merge = h$merge, height = h$height, order = h$order, cut = sapply(2:6, function(k) cutree(h, k))) }
# normalmixEM as copykat calls it
set.seed(3)
v <- list(c(rnorm(700, 0, 0.05), rnorm(300, 0.18, 0.06)), c(rnorm(500, -0.1, 0.04), rnorm(500, 0.05, 0.05)), rnorm(800, 0, 0.08))
out$em <- lapply(v, function(z) { s <- max(c(0.05, 0.5 * sd(z))); g <- capture.output(f <- normalmixEM(z, lambda = rep(1, 3)/3, mu = c(-0.2, 0, 0.2), sigma = s, arbvar = FALSE, ECM = FALSE, maxit = 5000)); list(x = z, sigma0 = s, sigma = f$sigma[1], mu = f$mu, lambda = f$lambda, loglik = f$loglik, iterations = length(f$all.loglik) - 1) })
# ks.test statistic with and without ties
set.seed(11); a <- rgamma(1000, 30, 31); b <- rgamma(1000, 33, 31)
out$ks <- list(a = a, b = b, stat = unname(suppressWarnings(ks.test(a, b))$statistic), a2 = round(a, 2), b2 = round(b, 2), stat2 = unname(suppressWarnings(ks.test(round(a, 2), round(b, 2)))$statistic))
writeLines(toJSON(out, digits = I(17), auto_unbox = TRUE), "data/r_components.json")
