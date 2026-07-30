"""R-parity tests for pycopykat against R copykat's exp.rawdata fixture.

Template-style offline parity test, per the omicverse-to-developer
conventions: an upstream R Rscript pre-writes TSVs under
``tests/r_out/``; this test loads those TSVs and asserts parity. If the
TSVs are absent (no R installed in CI), the test skips instead of
failing — the R reference is regenerated on a developer machine via::

    Rscript tests/r_reference.R

Bit-exact tier:
    * prediction row count matches R's prediction row count
    * prediction cell-ID set matches R's cell-ID set

Empirical tier:
    * per-cell label agreement (ARI / FMI) beats loose floors, since
      pycopykat's GMM baseline + ward.D2 linkage + dynamicTreeCut V1 do
      not reproduce R copykat bit-exactly (see Parity status in
      README.md). Floors are the same thresholds used by
      test_regression.py, conservative to allow one aneuploid/diploid
      label flip.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadr
import pytest
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score

from pycopykat import CopykatConfig, copykat


HERE = Path(__file__).parent
R_OUT = HERE / "r_out"

REF_RDA = Path("/media/jason/T7/rebuild/copykat-R/data/exp.rawdata.rda")

PRED_TSV = R_OUT / "exp_rawdata_prediction.tsv"
META_TSV = R_OUT / "exp_rawdata_meta.tsv"


def _require_r_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [p for p in (PRED_TSV, META_TSV) if not p.exists()]
    if missing:
        pytest.skip(
            "R reference not generated; run `Rscript tests/r_reference.R` "
            f"to produce: {[str(p) for p in missing]}"
        )
    return pd.read_csv(PRED_TSV, sep="\t"), pd.read_csv(META_TSV, sep="\t")


def _require_fixture() -> pd.DataFrame:
    if not REF_RDA.exists():
        pytest.skip(f"upstream fixture not present: {REF_RDA}")
    raw = next(iter(pyreadr.read_r(str(REF_RDA)).values()))
    assert isinstance(raw, pd.DataFrame)
    return raw


@pytest.fixture(scope="module")
def r_ref() -> pd.DataFrame:
    pred, _ = _require_r_reference()
    return pred


@pytest.fixture(scope="module")
def r_meta() -> pd.DataFrame:
    _, meta = _require_r_reference()
    return meta


@pytest.fixture(scope="module")
def py_pred() -> pd.DataFrame:
    _require_r_reference()  # skip early if R reference missing
    counts = _require_fixture()
    cfg = CopykatConfig(sam_name="exp_rawdata", n_jobs=1)
    result = copykat(counts, config=cfg)
    pred = result.prediction.copy()
    # Normalise column names to R copykat's schema.
    if "cell.names" not in pred.columns:
        pred = pred.reset_index().rename(columns={"index": "cell.names"})
    if "copykat.pred" not in pred.columns:
        for cand in ("prediction", "pred", "label"):
            if cand in pred.columns:
                pred = pred.rename(columns={cand: "copykat.pred"})
                break
    assert {"cell.names", "copykat.pred"}.issubset(pred.columns), pred.columns
    return pred[["cell.names", "copykat.pred"]]


# Tier 1 — API surface.
def test_copykat_callable() -> None:
    assert callable(copykat)


# Tier 2 — output shape (bit-exact).
def test_prediction_row_count_matches_r(py_pred: pd.DataFrame, r_ref: pd.DataFrame) -> None:
    assert len(py_pred) == len(r_ref), (
        f"prediction row count: py={len(py_pred)} R={len(r_ref)}"
    )


def test_prediction_row_count_matches_r_meta(r_ref: pd.DataFrame, r_meta: pd.DataFrame) -> None:
    meta_map = dict(zip(r_meta["key"], r_meta["value"]))
    assert int(meta_map["n_pred_rows"]) == len(r_ref)


# Tier 3 — output identity (bit-exact cell-ID set).
def test_cell_id_set_matches_r(py_pred: pd.DataFrame, r_ref: pd.DataFrame) -> None:
    py_ids = set(py_pred["cell.names"].astype(str))
    r_ids = set(r_ref["cell.names"].astype(str))
    only_py = py_ids - r_ids
    only_r = r_ids - py_ids
    assert not only_py and not only_r, (
        f"cell-ID set differs: only_py={len(only_py)} only_r={len(only_r)}"
    )


# Tier 4 — label values (empirical agreement).
def test_prediction_labels_agree_empirically(
    py_pred: pd.DataFrame, r_ref: pd.DataFrame
) -> None:
    merged = py_pred.merge(
        r_ref, on="cell.names", how="inner", suffixes=("_py", "_r")
    )
    assert len(merged) == len(r_ref), "inner merge should align all cells"

    ari = adjusted_rand_score(merged["copykat.pred_r"], merged["copykat.pred_py"])
    fmi = fowlkes_mallows_score(merged["copykat.pred_r"], merged["copykat.pred_py"])
    # Loose empirical floors — consistent with test_regression.py, allow
    # one aneuploid/diploid label flip on the exp.rawdata fixture.
    assert ari >= 0.80, f"py↔R ARI too low: {ari:.3f}"
    assert fmi >= 0.85, f"py↔R FMI too low: {fmi:.3f}"
