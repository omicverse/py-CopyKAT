"""pycopykat: a Python implementation of CopyKAT (Gao et al. 2021) that reproduces R copykat."""
from pycopykat.config import CopykatConfig
from pycopykat.io.h5ad import copykat_by_batch
from pycopykat.pipeline import copykat
from pycopykat.result import CopykatResult

__all__ = ["CopykatConfig", "CopykatResult", "copykat", "copykat_by_batch"]
__version__ = "0.2.0"
