"""Public API for baseline (diploid) estimation modes."""
from pycopykat.baseline._shared import BaselineResult, ward_cluster_with_min_size
from pycopykat.baseline.auto import baseline_norm_cl
from pycopykat.baseline.gmm import baseline_gmm
from pycopykat.baseline.synthetic import SyntheticBaselineResult, baseline_synthetic

__all__ = [
    "BaselineResult",
    "SyntheticBaselineResult",
    "baseline_gmm",
    "baseline_norm_cl",
    "baseline_synthetic",
    "ward_cluster_with_min_size",
]
