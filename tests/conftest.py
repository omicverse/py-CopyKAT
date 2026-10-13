"""Shared pytest fixtures for the pycopykat test-suite."""
from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable

import pytest


@pytest.fixture
def require_r_package() -> Callable[[str], None]:
    """Return a callable that skips the current test unless R + ``package`` are usable.

    A few kernel tests cross-check the Python kernels against their R
    counterparts (``dlm::dlmSmooth``, ``MCMCpack::MCpoissongamma``) by
    shelling out to ``Rscript``. R is a developer-only, optional toolchain:
    the CI runners have no R at all, and a developer's R may lack the
    package. Either way the test is skipped with a clear reason rather than
    erroring, mirroring how ``tests/test_r_parity.py`` skips when its R
    reference outputs are absent.
    """

    def _require(package: str) -> None:
        if shutil.which("Rscript") is None:
            pytest.skip(f"Rscript not on PATH (R parity test needs R package {package!r})")
        probe = subprocess.run(
            [
                "Rscript",
                "-e",
                f'quit(status = if (requireNamespace("{package}", quietly = TRUE)) 0L else 42L)',
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 42:
            pytest.skip(f"R package {package!r} not installed")
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout).strip()
            pytest.skip(f"could not probe R for package {package!r}: {detail}")

    return _require
