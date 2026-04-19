"""End-to-end regression: Python copykat vs R copykat on exp.rawdata.rda.

Marked ``slow``; excluded from the default unit-test suite. Runs the full
R driver + py pipeline on the ~400-cell reference dataset bundled with
copykat and asserts that:

* prediction ARI ≥ 0.85  (one-of-20 flip corresponds to ARI ≈ 0.80)
* prediction kappa ≥ 0.85
* CNA Spearman median ≥ 0.90

Thresholds are intentionally looser than the R↔R reference so that
algorithmic divergences (ward.D vs ward.D2, mixtools vs sklearn GMM,
dynamicTreeCut V1) are accepted — M7.5 is the debugging driver, not a
bit-exact gate.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadr
import pytest

from pycopykat import CopykatConfig, copykat
from pycopykat.validation.metrics import compare_cna, compare_predictions
from pycopykat.validation.r_runner import load_r_cna, load_r_prediction, run_r_copykat


REF_RDA = Path("/media/jason/T7/rerbulid/copykat-R/data/exp.rawdata.rda")


@pytest.mark.slow
def test_regression_exp_rawdata(tmp_path):
    if not REF_RDA.exists():
        pytest.skip(f"reference rda not present: {REF_RDA}")

    mat = next(iter(pyreadr.read_r(str(REF_RDA)).values()))
    assert isinstance(mat, pd.DataFrame), f"expected DataFrame, got {type(mat)}"

    r_out = tmp_path / "r"
    run_r_copykat(mat, r_out, sam_name="reg", n_cores=4)
    pred_r = load_r_prediction(r_out, "reg")
    cna_r = load_r_cna(r_out, "reg")

    cfg = CopykatConfig(
        n_jobs=4, sam_name="py", output_dir=tmp_path / "py"
    )
    res_py = copykat(mat, config=cfg)
    pred_py = res_py.prediction
    cna_py = res_py.cna_mat

    pred_cmp = compare_predictions(pred_r, pred_py)
    cna_cmp = compare_cna(
        cna_r, cna_py, method="spearman",
    )
    print(f"\n=== regression metrics ===")
    print(f"py_vs_r predictions: {pred_cmp}")
    print(f"py_vs_r cna:         {cna_cmp}")

    # Loose acceptance thresholds — see module docstring.
    assert pred_cmp["ari"] >= 0.85, f"ARI {pred_cmp['ari']:.3f} < 0.85"
    assert pred_cmp["kappa"] >= 0.85, f"kappa {pred_cmp['kappa']:.3f} < 0.85"
    assert cna_cmp["median_r"] >= 0.90, (
        f"Spearman median {cna_cmp['median_r']:.3f} < 0.90"
    )
