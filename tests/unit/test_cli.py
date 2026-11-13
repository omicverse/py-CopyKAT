"""Tests for pycopykat.cli — typer CLI."""
import numpy as np
import pandas as pd
from typer.testing import CliRunner

from pycopykat.cli import app

runner = CliRunner()


def test_cli_help():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "run" in res.output


def test_cli_run_on_csv(tmp_path):
    """The CLI on a CSV of copykat's example data (first 150 cells): same calls as the API."""
    from pathlib import Path

    from scipy import sparse

    from pycopykat import CopykatConfig, copykat

    z = np.load(Path(__file__).parents[1] / "data" / "exp_rawdata.npz")
    X = sparse.csc_matrix((z["data"].astype(float), z["indices"].astype(np.int64), z["indptr"]),
                          shape=tuple(z["shape"]))[:, :150]
    mat = pd.DataFrame(X.toarray(), index=z["genes"].astype(str), columns=z["cells"].astype(str)[:150])
    mat = mat.loc[mat.sum(axis=1) > 0]
    csv = tmp_path / "in.csv"
    mat.to_csv(csv)
    out = tmp_path / "out"
    res = runner.invoke(app, ["run", "--input", str(csv), "--output-dir", str(out), "--sam-name", "cli"])
    assert res.exit_code == 0, res.output
    pred = pd.read_csv(out / "cli_copykat_prediction.txt", sep="\t")
    api = copykat(mat, CopykatConfig())
    assert pred.equals(api.prediction)
    assert (out / "cli_copykat_CNA_results.txt").exists()
