"""Smoke tests — verify the package imports and its public API surface
resolves without R installed. Runs in CI in seconds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import pycopykat


def test_version_is_set() -> None:
    assert isinstance(pycopykat.__version__, str)
    assert pycopykat.__version__  # non-empty


def test_public_api_imports() -> None:
    from pycopykat import CopykatConfig, CopykatResult, copykat

    assert callable(copykat)
    assert CopykatConfig is not None
    assert CopykatResult is not None


def test_config_defaults_instantiate() -> None:
    cfg = pycopykat.CopykatConfig()
    assert cfg is not None


def test_anndata_batch_driver_importable() -> None:
    from pycopykat import copykat_by_batch

    assert callable(copykat_by_batch)


def test_counts_dataframe_contract() -> None:
    """Smoke-check the genes × cells DataFrame shape pycopykat consumes."""
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(
        rng.integers(0, 10, size=(50, 8)),
        index=[f"gene{i}" for i in range(50)],
        columns=[f"cell{j}" for j in range(8)],
    )
    assert counts.shape == (50, 8)
    assert counts.index.name is None
    assert (counts.values >= 0).all()
