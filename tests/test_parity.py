"""pycopykat against R copykat 1.2.5 on copykat's example data (``exp.rawdata``, 302 cells).

The R results come from ``tests/r_reference.R`` (``n.cores = 1``); the input is packed in
``tests/data/exp_rawdata.npz``. With the same seed the two give the same breakpoints, the same
pre-defined diploid cells and the same call for every cell; the copy-number matrices agree to
about 1e-10 (R's dlm smoother works on SVD factors).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from pycopykat import CopykatConfig, core
from pycopykat.pipeline import _copykat_from_sparse

D = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def exp_rawdata():
    z = np.load(D / "exp_rawdata.npz")
    X = sparse.csc_matrix((z["data"].astype(np.float64), z["indices"].astype(np.int64), z["indptr"]),
                          shape=tuple(z["shape"]))
    return X, z["genes"].astype(str), z["cells"].astype(str)


@pytest.fixture(scope="module")
def run(exp_rawdata):
    X, genes, cells = exp_rawdata
    return core.run(X, genes, cells, keep_intermediate=True)


def test_breakpoints_identical(run):
    r = np.loadtxt(D / "r_exp_breaks.txt", dtype=np.int64)
    assert np.array_equal(run.intermediate["breaks"], r)


def test_baseline_cells_identical(run):
    assert run.intermediate["bnc_preN"] == (D / "r_exp_preN.txt").read_text().split()


def test_prediction_identical(run):
    r = pd.read_csv(D / "r_exp_prediction.tsv", sep="\t")
    assert r.equals(run.prediction)


def test_cna_matches(run):
    cna = run.cna.iloc[:, 3:].to_numpy()
    col = np.loadtxt(D / "r_exp_cna_colsums.txt")
    row = np.loadtxt(D / "r_exp_cna_rowsums.txt")
    assert np.allclose(cna.sum(axis=0), col, rtol=0, atol=1e-6)
    assert np.allclose(cna.sum(axis=1), row, rtol=0, atol=1e-6)


def test_public_api_and_threads(exp_rawdata):
    X, genes, cells = exp_rawdata
    r = pd.read_csv(D / "r_exp_prediction.tsv", sep="\t")
    res = _copykat_from_sparse(X, genes, cells, CopykatConfig(n_jobs=2))
    assert r.equals(res.prediction)
    assert res.cna_mat.index.names == ["chrom", "chrompos", "abspos"]
    assert list(res.cna_mat.columns) == list(r["cell.names"][r["copykat.pred"] != "not.defined"])
