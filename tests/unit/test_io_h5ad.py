"""Unit tests for the h5ad batch-split driver.

The full copykat pipeline is monkey-patched to a tiny stub so these tests
verify the *merge* logic (obs/obsm/uns attachment, batch skipping, error
handling) rather than the numerical pipeline itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata")

from pycopykat.config import CopykatConfig  # noqa: E402
from pycopykat.io import h5ad as h5ad_mod  # noqa: E402
from pycopykat.result import CopykatResult  # noqa: E402


def _fake_result_factory(batch_to_pred: dict[str, dict[str, str]]):
    """Return a fake copykat() that reads the incoming counts' cell columns and
    produces a CopykatResult with predictions drawn from ``batch_to_pred``.

    Each call is keyed by the sam_name (which the driver sets to ``<base>_<batch>``).
    """

    def _fake(mat: pd.DataFrame, config: CopykatConfig | None = None, **_: object) -> CopykatResult:
        sam = config.sam_name if config else ""
        # the sam_name carries the batch label after the underscore
        batch_label = sam.rsplit("_", 1)[-1] if "_" in sam else sam
        pred_map = batch_to_pred[batch_label]
        cells = [c for c in mat.columns if c in pred_map]
        pred_df = pd.DataFrame(
            {
                "cell": cells,
                "copykat.pred": [pred_map[c] for c in cells],
            }
        )
        n_bins = 4  # dummy tiny bin count
        cna = pd.DataFrame(
            np.full((n_bins, len(cells)), 0.1, dtype=np.float64),
            index=pd.MultiIndex.from_tuples(
                [(1, 0, 1e5), (1, 1e5, 2e5), (2, 0, 1e5), (2, 1e5, 2e5)],
                names=["chrom", "start", "end"],
            ),
            columns=cells,
        )
        subclone = pd.Series(
            [1] * sum(1 for c in cells if pred_map[c] == "aneuploid"),
            index=[c for c in cells if pred_map[c] == "aneuploid"],
            name="subclone",
        )
        return CopykatResult(
            cna_mat=cna,
            prediction=pred_df,
            linkage=np.zeros((max(len(cells) - 1, 0), 4), dtype=np.float64),
            subclone=subclone,
            warnings=("note: fake result",),
        )

    return _fake


def _make_tiny_adata(n_cells: int = 12, n_genes: int = 30) -> "anndata.AnnData":
    rng = np.random.default_rng(0)
    X = rng.poisson(3, size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "patient": (["P1"] * (n_cells // 2)) + (["P2"] * (n_cells - n_cells // 2)),
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)])
    return anndata.AnnData(X=X, obs=obs, var=var)


def test_copykat_by_batch_merges_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    ad = _make_tiny_adata(n_cells=12)
    p1_cells = ad.obs.index[ad.obs["patient"] == "P1"].tolist()
    p2_cells = ad.obs.index[ad.obs["patient"] == "P2"].tolist()
    batch_to_pred = {
        "P1": {c: "diploid" for c in p1_cells},
        "P2": {c: ("aneuploid" if i % 2 == 0 else "diploid") for i, c in enumerate(p2_cells)},
    }
    monkeypatch.setattr(h5ad_mod, "copykat", _fake_result_factory(batch_to_pred))
    # Replace the heavy bin loader with a tiny 4-bin stub matching the fake CNA rows
    monkeypatch.setattr(
        h5ad_mod,
        "load_hg20_bins",
        lambda: pd.DataFrame(
            {
                "chrom": [1, 1, 2, 2, 24],  # 24 is dropped by the filter
                "chrompos": [1e5, 2e5, 1e5, 2e5, 1e5],
            }
        ),
    )

    cfg = CopykatConfig(sam_name="cohort", n_jobs=1)
    out = h5ad_mod.copykat_by_batch(
        ad, batch_key="patient", config=cfg, min_cells_per_batch=2
    )

    # predictions attach for every cell
    assert "copykat_pred" in out.obs
    for c in p1_cells:
        assert out.obs.loc[c, "copykat_pred"] == "diploid"
    for i, c in enumerate(p2_cells):
        want = "aneuploid" if i % 2 == 0 else "diploid"
        assert out.obs.loc[c, "copykat_pred"] == want
    # subclone labels only on aneuploid cells
    aneu_cells = [c for i, c in enumerate(p2_cells) if i % 2 == 0]
    for c in aneu_cells:
        assert out.obs.loc[c, "copykat_subclone"] == 1
    for c in p1_cells:
        assert pd.isna(out.obs.loc[c, "copykat_subclone"])
    # obsm matrix present, shape (n_cells, n_bins_retained)
    assert "copykat_cna" in out.obsm
    assert out.obsm["copykat_cna"].shape == (ad.n_obs, 4)
    # bin table stored, excluding chrom 24
    assert "copykat_bins" in out.uns
    assert set(out.uns["copykat_bins"]["chrom"].tolist()) == {1, 2}
    # per-batch metadata
    meta = out.uns["copykat_per_batch"]
    assert meta["P1"]["status"] == "ok"
    assert meta["P2"]["status"] == "ok"
    assert meta["P1"]["n_cells_out"] == len(p1_cells)


def test_copykat_by_batch_skips_small_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    ad = _make_tiny_adata(n_cells=12)
    # min_cells_per_batch high enough to skip both batches
    monkeypatch.setattr(h5ad_mod, "copykat", _fake_result_factory({}))
    monkeypatch.setattr(
        h5ad_mod,
        "load_hg20_bins",
        lambda: pd.DataFrame({"chrom": [1], "chrompos": [1e5]}),
    )
    out = h5ad_mod.copykat_by_batch(
        ad, batch_key="patient", config=CopykatConfig(n_jobs=1), min_cells_per_batch=1000
    )
    meta = out.uns["copykat_per_batch"]
    assert all(v["status"] == "skipped_too_small" for v in meta.values())
    # all predictions remain 'not.defined'
    assert (out.obs["copykat_pred"] == "not.defined").all()


def test_copykat_by_batch_on_error_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    ad = _make_tiny_adata(n_cells=12)

    def _raising(mat, config=None, **_):
        raise RuntimeError("boom")

    monkeypatch.setattr(h5ad_mod, "copykat", _raising)
    monkeypatch.setattr(
        h5ad_mod,
        "load_hg20_bins",
        lambda: pd.DataFrame({"chrom": [1], "chrompos": [1e5]}),
    )
    out = h5ad_mod.copykat_by_batch(
        ad, batch_key="patient", config=CopykatConfig(n_jobs=1),
        min_cells_per_batch=2, on_error="skip",
    )
    meta = out.uns["copykat_per_batch"]
    for v in meta.values():
        assert v["status"] == "error"
        assert "RuntimeError" in v["error"]


def test_copykat_by_batch_bad_batch_key() -> None:
    ad = _make_tiny_adata(n_cells=4)
    with pytest.raises(KeyError):
        h5ad_mod.copykat_by_batch(ad, batch_key="missing", config=CopykatConfig())
