"""Smoke test: full copykat() pipeline on tiny synthetic data."""
import numpy as np
import pandas as pd

from pycopykat import CopykatConfig, copykat
from pycopykat.io.annotation import load_hg20_annotation


def test_copykat_runs_on_tiny_data():
    rng = np.random.default_rng(0)
    ann = load_hg20_annotation()
    # Take 400 real hg20 HGNC symbols spanning several chromosomes
    syms_pool = (
        ann.dropna(subset=["hgnc_symbol"])
        .drop_duplicates("hgnc_symbol")
        .sort_values("abspos")
    )
    # Pick genes from each of chroms 1..10 for cross-chrom coverage
    picks = (
        syms_pool.groupby("chromosome_name", sort=False)
        .head(80)
        .loc[syms_pool["chromosome_name"].isin(range(1, 11))]
    )
    genes = picks["hgnc_symbol"].unique().tolist()[:500]

    n_cells = 60
    mat = pd.DataFrame(
        rng.poisson(5, size=(len(genes), n_cells)).astype(int),
        index=genes,
        columns=[f"c{i}" for i in range(n_cells)],
    )
    cfg = CopykatConfig(
        min_gene_per_cell=10,
        low_dr=0.01,
        up_dr=0.02,
        win_size=10,
        ks_cut=0.3,
        ngene_chr=1,
        n_jobs=1,
        sam_name="tiny",
    )
    res = copykat(mat, config=cfg)
    assert res.prediction.shape[0] >= 10
    classes = set(res.prediction["copykat.pred"].unique())
    # Accept both plain and low-confidence labels
    allowed = {
        "diploid", "aneuploid",
        "c1:diploid:low.conf", "c2:aneuploid:low.conf",
    }
    assert classes.issubset(allowed), f"unexpected labels: {classes}"
    assert res.cna_mat.shape[1] == res.prediction.shape[0]
