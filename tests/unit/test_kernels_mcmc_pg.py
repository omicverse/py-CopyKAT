import subprocess

import numpy as np
import pytest
from scipy.stats import ks_2samp

from pycopykat.kernels.mcmc_pg import pg_posterior_mean, pg_posterior_samples


def _require_r_package(package: str) -> None:
    try:
        result = subprocess.run(
            [
                "Rscript",
                "-e",
                f'if (!requireNamespace("{package}", quietly=TRUE)) quit(status=42)',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pytest.skip("Rscript not available")

    if result.returncode == 42:
        pytest.skip(f"R package {package!r} not installed")
    if result.returncode != 0:
        msg = (result.stderr or result.stdout).strip()
        pytest.skip(f"R package check failed for {package!r}: {msg}")


def test_posterior_samples_shape():
    rng = np.random.default_rng(42)
    y = rng.poisson(5.0, size=30).astype(np.float64)
    samp = pg_posterior_samples(y, alpha=1.0, beta=1.0, mc=1000, seed=0)
    assert samp.shape == (1000,)
    expected = (1.0 + y.sum()) / (1.0 + y.size)
    assert abs(samp.mean() - expected) < 0.5


def test_matches_mcmcpack(tmp_path):
    _require_r_package("MCMCpack")
    rng = np.random.default_rng(42)
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
    stat, p = ks_2samp(r_samp, py_samp)
    assert p > 0.01, f"KS p={p:.3f} too small (stat={stat:.3f})"


def test_mean_batched():
    rng = np.random.default_rng(42)
    y_segments = [rng.poisson(rate, size=20).astype(np.float64) for rate in (1, 5, 20)]
    means = np.array([
        pg_posterior_mean(y, 1.0, 1.0, mc=1000, seed=i)
        for i, y in enumerate(y_segments)
    ])
    analytic = np.array([(1.0 + y.sum()) / (1.0 + y.size) for y in y_segments])
    np.testing.assert_allclose(means, analytic, rtol=0.1)
