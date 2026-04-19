"""Tests for pycopykat.cli — typer CLI."""
import numpy as np
import pandas as pd
from typer.testing import CliRunner

from pycopykat.cli import app
from pycopykat.io.annotation import load_hg20_annotation

runner = CliRunner()


def test_cli_help():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "run" in res.output


def test_cli_run_on_csv(tmp_path):
    ann = load_hg20_annotation()
    picks = (
        ann.dropna(subset=["hgnc_symbol"])
        .drop_duplicates("hgnc_symbol")
        .loc[ann["chromosome_name"].isin(range(1, 11))]
        .groupby("chromosome_name", sort=False)
        .head(50)
    )
    genes = picks["hgnc_symbol"].unique().tolist()[:300]
    mat = pd.DataFrame(
        np.random.default_rng(0).poisson(5, size=(len(genes), 40)),
        index=genes,
        columns=[f"c{i}" for i in range(40)],
    )
    csv = tmp_path / "in.csv"
    mat.to_csv(csv)
    out = tmp_path / "out"
    res = runner.invoke(
        app,
        [
            "run",
            "--input", str(csv),
            "--output-dir", str(out),
            "--sam-name", "cli",
            "--min-gene-per-cell", "5",
            "--low-dr", "0.01",
            "--up-dr", "0.02",
            "--win-size", "10",
            "--ks-cut", "0.3",
            "--ngene-chr", "1",
        ],
    )
    assert res.exit_code == 0, res.output
    assert (out / "cli_copykat_prediction.txt").exists()
