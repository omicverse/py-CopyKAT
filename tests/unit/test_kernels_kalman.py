# ruff: noqa: N803, N806  # matrix-style uppercase names (dW/dV/Y) by convention
import subprocess

import numpy as np
import pytest

from pycopykat.kernels.kalman import kalman_smooth, kalman_smooth_matrix

rng = np.random.default_rng(0)


def _simulate(n=200, dW=0.001, dV=0.16, seed=0):
    r = np.random.default_rng(seed)
    x = np.cumsum(r.normal(0, np.sqrt(dW), size=n))
    y = x + r.normal(0, np.sqrt(dV), size=n)
    return y


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


def test_smoother_shape():
    y = _simulate()
    s = kalman_smooth(y, dV=0.16, dW=0.001)
    assert s.shape == y.shape


def test_smoother_reduces_variance():
    y = _simulate(n=500)
    s = kalman_smooth(y, dV=0.16, dW=0.001)
    # Smoothed series should have much smaller first-difference variance
    assert np.var(np.diff(s)) < np.var(np.diff(y)) * 0.5


def test_matches_r_dlm_smooth(tmp_path):
    """Bit-close match to dlm::dlmSmooth on same input."""
    _require_r_package("dlm")
    y = _simulate(n=50, seed=42)
    rscript = tmp_path / "run.R"
    ypath = tmp_path / "y.txt"
    opath = tmp_path / "out.txt"
    np.savetxt(ypath, y)
    rscript.write_text(f"""
      library(dlm)
      y <- scan("{ypath}", quiet=TRUE)
      m <- dlmModPoly(order=1, dV=0.16, dW=0.001)
      s <- dlmSmooth(y, m)$s
      s <- s[2:length(s)]
      s <- s - mean(s)
      write.table(s, "{opath}", row.names=FALSE, col.names=FALSE)
    """)
    subprocess.run(["Rscript", str(rscript)], check=True, capture_output=True)
    r_s = np.loadtxt(opath)
    py_s = kalman_smooth(y, dV=0.16, dW=0.001)
    py_s = py_s - py_s.mean()  # R copykat re-centers after smoothing
    np.testing.assert_allclose(py_s, r_s, rtol=1e-4, atol=1e-4)


def test_matrix_version_equivalent_to_loop():
    Y = rng.standard_normal((100, 20))
    loop = np.column_stack([kalman_smooth(Y[:, j], 0.16, 0.001) for j in range(20)])
    mat = kalman_smooth_matrix(Y, 0.16, 0.001)
    np.testing.assert_allclose(mat, loop, rtol=1e-10, atol=1e-12)
