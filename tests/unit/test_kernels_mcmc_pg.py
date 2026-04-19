import subprocess

import numpy as np
from scipy.stats import ks_2samp

from pycopykat.kernels.mcmc_pg import pg_posterior_mean, pg_posterior_samples

rng = np.random.default_rng(42)


def test_posterior_samples_shape():
    y = rng.poisson(5.0, size=30).astype(np.float64)
    samp = pg_posterior_samples(y, alpha=1.0, beta=1.0, mc=1000, seed=0)
    assert samp.shape == (1000,)
    # Analytic posterior mean = (a + sum(y)) / (b + n)
    expected = (1.0 + y.sum()) / (1.0 + y.size)
    assert abs(samp.mean() - expected) < 0.5


def test_matches_mcmcpack(tmp_path):
    y = rng.poisson(5.0, size=40).astype(np.float64)
    np.savetxt(tmp_path / "y.txt", y)
    rscript = tmp_path / "run.R"
    rscript.write_text(f"""
      suppressMessages(library(MCMCpack))
      y <- scan("{tmp_path}/y.txt", quiet=TRUE)
      set.seed(123)
      s <- as.numeric(MCpoissongamma(y, 1.0, 1.0, mc=2000))
      write.table(s, "{tmp_path}/s.txt", row.names=FALSE, col.names=FALSE)
    """)
    subprocess.run(["Rscript", str(rscript)], check=True, capture_output=True)
    r_samp = np.loadtxt(tmp_path / "s.txt")
    py_samp = pg_posterior_samples(y, alpha=1.0, beta=1.0, mc=2000, seed=123)
    # Two i.i.d. Gamma samples of same size should have similar distributions
    stat, p = ks_2samp(r_samp, py_samp)
    assert p > 0.01, f"KS p={p:.3f} too small (stat={stat:.3f})"


def test_mean_batched():
    y_segments = [
        rng.poisson(rate, size=20).astype(np.float64) for rate in (1, 5, 20)
    ]
    means = np.array(
        [
            pg_posterior_mean(y, 1.0, 1.0, mc=1000, seed=i)
            for i, y in enumerate(y_segments)
        ]
    )
    analytic = np.array([(1.0 + y.sum()) / (1.0 + y.size) for y in y_segments])
    np.testing.assert_allclose(means, analytic, rtol=0.1)
