"""Tests for pycopykat.viz.heatmap — CNA heatmap PNG output."""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pycopykat.viz.heatmap import (  # noqa: E402
    plot_cna_delta,
    plot_cna_heatmap,
    plot_cna_heatmap_compare,
)


def test_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    cna = pd.DataFrame(
        rng.standard_normal((200, 30)),
        index=pd.MultiIndex.from_arrays(
            [[1] * 100 + [2] * 100, np.arange(200)], names=["chrom", "bin"]
        ),
        columns=[f"c{i}" for i in range(30)],
    )
    pred = pd.DataFrame({
        "cell": cna.columns,
        "copykat.pred": ["aneuploid" if i % 2 else "diploid" for i in range(30)],
    })
    out = tmp_path / "heatmap.png"
    plot_cna_heatmap(cna, pred, output=out)
    assert out.exists()
    assert out.stat().st_size > 5_000


def test_handles_plain_index(tmp_path):
    rng = np.random.default_rng(1)
    cna = pd.DataFrame(
        rng.standard_normal((80, 15)),
        columns=[f"c{i}" for i in range(15)],
    )
    pred = pd.DataFrame({
        "cell": cna.columns,
        "copykat.pred": ["diploid"] * 15,
    })
    out = tmp_path / "heatmap.png"
    plot_cna_heatmap(cna, pred, output=out)
    assert out.exists()


def _toy_compare_inputs(seed: int = 2):
    rng = np.random.default_rng(seed)
    bin_index = pd.MultiIndex.from_arrays(
        [[1] * 60 + [2] * 60, np.arange(120)], names=["chrom", "bin"]
    )
    cells = [f"c{i}" for i in range(20)]
    py_cna = pd.DataFrame(rng.standard_normal((120, 20)), index=bin_index, columns=cells)
    r_cna = py_cna + rng.standard_normal(py_cna.shape) * 0.1
    py_pred = pd.DataFrame({
        "cell": cells,
        "copykat.pred": ["aneuploid" if i % 2 else "diploid" for i in range(20)],
    })
    r_pred = pd.DataFrame({
        "cell.names": cells,
        "copykat.pred": ["aneuploid" if i < 12 else "diploid" for i in range(20)],
    })
    return py_cna, r_cna, py_pred, r_pred


def test_compare_returns_two_axes():
    py_cna, r_cna, py_pred, r_pred = _toy_compare_inputs()
    ax_py, ax_r = plot_cna_heatmap_compare(py_cna, r_cna, py_pred, r_pred)
    assert ax_py.get_title() == "pycopykat"
    assert ax_r.get_title() == "R copykat"
    plt.close("all")


def test_delta_runs_with_and_without_pred():
    py_cna, r_cna, py_pred, _ = _toy_compare_inputs(3)
    ax = plot_cna_delta(py_cna, r_cna, py_pred)
    assert "Δ CNA" in ax.get_title()
    plt.close("all")
    ax = plot_cna_delta(py_cna, r_cna)
    assert "Δ CNA" in ax.get_title()
    plt.close("all")
