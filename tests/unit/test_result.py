import numpy as np
import pandas as pd

from pycopykat.result import CopykatResult


def _toy_result():
    cna = pd.DataFrame(
        np.zeros((5, 3)),
        columns=["c1", "c2", "c3"],
        index=pd.MultiIndex.from_tuples(
            [(1, 100, 200), (1, 200, 300), (1, 300, 400), (2, 100, 200), (2, 200, 300)],
            names=["chrom", "start", "end"],
        ),
    )
    pred = pd.DataFrame(
        {"cell": ["c1", "c2", "c3"], "copykat.pred": ["aneuploid", "diploid", "diploid"]}
    )
    return CopykatResult(
        cna_mat=cna,
        prediction=pred,
        linkage=np.zeros((2, 4)),
        subclone=pd.Series(["c1"], index=["c1"]),
        warnings=("ok",),
    )


def test_result_has_expected_attrs():
    r = _toy_result()
    assert r.cna_mat.shape == (5, 3)
    assert r.prediction["copykat.pred"].tolist() == ["aneuploid", "diploid", "diploid"]
    assert r.warnings == ("ok",)


def test_to_txt_writes_r_compatible_files(tmp_path):
    r = _toy_result()
    r.to_txt(tmp_path, "sam")
    cna = pd.read_csv(tmp_path / "sam_copykat_CNA_results.txt", sep="\t")
    pred = pd.read_csv(tmp_path / "sam_copykat_prediction.txt", sep="\t")
    assert {"chrom", "start", "end", "c1", "c2", "c3"}.issubset(cna.columns)
    assert len(cna) == 5
    assert pred["copykat.pred"].tolist() == ["aneuploid", "diploid", "diploid"]


def test_to_h5ad_roundtrip(tmp_path):
    import anndata as ad

    r = _toy_result()
    path = tmp_path / "out.h5ad"
    r.to_h5ad(path)
    adata = ad.read_h5ad(path)
    assert adata.n_obs == 3  # cells
    assert adata.n_vars == 5  # bins
    assert "copykat.pred" in adata.obs.columns
