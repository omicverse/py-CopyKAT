from pycopykat.config import CopykatConfig


def test_default_config():
    c = CopykatConfig()
    assert c.id_type == "Symbol"
    assert c.genome == "hg20"
    assert c.ngene_chr == 5
    assert c.min_gene_per_cell == 200
    assert c.low_dr == 0.05
    assert c.up_dr == 0.1
    assert c.win_size == 25
    assert c.ks_cut == 0.1
    assert c.distance == "euclidean"
    assert c.cell_line is False
    assert c.seed == 1234
    assert c.n_jobs == 1


def test_config_accepts_overrides():
    c = CopykatConfig(n_jobs=8, distance="pearson")
    assert c.n_jobs == 8
    assert c.distance == "pearson"


def test_config_rejects_invalid_distance():
    import pytest

    with pytest.raises(ValueError):
        CopykatConfig(distance="manhattan")  # not supported
