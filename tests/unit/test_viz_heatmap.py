"""Tests for pycopykat.viz.heatmap — CNA heatmap PNG output."""
import numpy as np
import pandas as pd

from pycopykat.viz.heatmap import plot_cna_heatmap


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
