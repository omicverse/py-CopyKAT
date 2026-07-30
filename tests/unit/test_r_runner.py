"""Tests for pycopykat.validation.r_runner — R prediction / CNA loaders."""
import pandas as pd
import pytest

from pycopykat.validation.r_runner import load_r_cna, load_r_prediction


def _write_fake_r_outputs(tmp_path, sam):
    pred = pd.DataFrame({
        "cell.names":  ["c0", "c1", "c2"],
        "copykat.pred": ["diploid", "aneuploid", "diploid"],
    })
    pred.to_csv(tmp_path / f"{sam}_copykat_prediction.txt",
                sep="\t", index=False)
    cna = pd.DataFrame({
        "chrom":    [1, 1, 2],
        "chrompos": [100, 200, 300],
        "abspos":   [100, 200, 400],
        "c0":       [0.0, 0.1, 0.2],
        "c1":       [0.3, 0.1, -0.1],
        "c2":       [0.0, 0.0, 0.0],
    })
    cna.to_csv(tmp_path / f"{sam}_copykat_CNA_results.txt",
               sep="\t", index=False)


def test_load_r_prediction_normalises_columns(tmp_path):
    _write_fake_r_outputs(tmp_path, "sam")
    df = load_r_prediction(tmp_path, "sam")
    assert list(df.columns) == ["cell", "copykat.pred"]
    assert df.shape == (3, 2)
    assert set(df["copykat.pred"]) == {"diploid", "aneuploid"}


def test_load_r_cna_sets_multiindex(tmp_path):
    _write_fake_r_outputs(tmp_path, "sam")
    df = load_r_cna(tmp_path, "sam")
    assert df.shape == (3, 3)
    assert df.index.names == ["chrom", "chrompos", "abspos"]
    assert set(df.columns) == {"c0", "c1", "c2"}


def test_load_r_prediction_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_r_prediction(tmp_path, "nonexistent")


def test_loaders_drop_unnamed_index_column(tmp_path):
    sam = "sam"
    pred = pd.DataFrame({
        "cell.names": ["c0", "c1"],
        "copykat.pred": ["diploid", "aneuploid"],
    })
    pred.to_csv(tmp_path / f"{sam}_copykat_prediction.txt", sep="\t", index=True)

    cna = pd.DataFrame({
        "chrom": [1, 1],
        "chrompos": [100, 200],
        "abspos": [100, 200],
        "c0": [0.0, 0.1],
        "c1": [0.2, -0.1],
    })
    cna.to_csv(tmp_path / f"{sam}_copykat_CNA_results.txt", sep="\t", index=True)

    pred_df = load_r_prediction(tmp_path, sam)
    cna_df = load_r_cna(tmp_path, sam)
    assert list(pred_df.columns) == ["cell", "copykat.pred"]
    assert pred_df.shape == (2, 2)
    assert cna_df.index.names == ["chrom", "chrompos", "abspos"]
    assert cna_df.shape == (2, 2)
