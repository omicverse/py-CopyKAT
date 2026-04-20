"""R vs Python comparison metrics for CopyKAT outputs.

* :func:`compare_predictions` — ARI / Cohen κ / FMI of two per-cell
  ``diploid``/``aneuploid`` prediction DataFrames.
* :func:`compare_cna` — per-cell Spearman (or Pearson) correlation between two
  bin × cell CNA matrices, reported as median / mean / min across cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    cohen_kappa_score,
    fowlkes_mallows_score,
)


def _normalise_pred_label(s: pd.Series) -> pd.Series:
    """Collapse R's low-confidence tags to their base label for comparison."""
    return s.astype(str).str.replace(
        r"^c[12]:(diploid|aneuploid):low\.conf$", r"\1", regex=True
    )


def compare_predictions(
    pred_r: pd.DataFrame, pred_py: pd.DataFrame
) -> dict[str, float | int]:
    """Compare two prediction DataFrames on their shared cell names.

    Both frames must have ``cell`` and ``copykat.pred`` columns. Only cells
    appearing in both are scored.
    """
    m = pred_r.merge(pred_py, on="cell", suffixes=("_r", "_py"))
    if m.empty:
        return {
            "n_shared": 0,
            "ari": float("nan"),
            "kappa": float("nan"),
            "fmi": float("nan"),
        }
    r = _normalise_pred_label(m["copykat.pred_r"])
    p = _normalise_pred_label(m["copykat.pred_py"])
    r_bin = (r == "aneuploid").astype(int).to_numpy()
    p_bin = (p == "aneuploid").astype(int).to_numpy()
    return {
        "n_shared": int(len(m)),
        "ari": float(adjusted_rand_score(r, p)),
        "kappa": float(cohen_kappa_score(r, p)),
        "fmi": float(fowlkes_mallows_score(r_bin, p_bin)),
    }


def compare_cna(
    cna_r: pd.DataFrame,
    cna_py: pd.DataFrame,
    *,
    method: str = "spearman",
) -> dict[str, float | int | str]:
    """Per-cell correlation between two bin × cell CNA matrices.

    Parameters
    ----------
    cna_r, cna_py
        DataFrames with cell names as columns.
    method
        ``"spearman"`` (default) or ``"pearson"``.

    Returns
    -------
    Dict with ``n_shared`` (cells), ``median_r``, ``mean_r``, ``min_r``,
    and echo of ``method``.
    """
    shared = sorted(set(cna_r.columns) & set(cna_py.columns))
    if not shared:
        return {
            "n_shared": 0,
            "method": method,
            "median_r": float("nan"),
            "mean_r": float("nan"),
            "min_r": float("nan"),
        }
    fn = spearmanr if method == "spearman" else pearsonr
    n = min(len(cna_r), len(cna_py))
    rs: list[float] = []
    for c in shared:
        r, _ = fn(cna_r[c].to_numpy()[:n], cna_py[c].to_numpy()[:n])
        rs.append(float(r) if r == r else 0.0)  # NaN → 0 (constant column)
    arr = np.asarray(rs)
    return {
        "n_shared": len(shared),
        "method": method,
        "median_r": float(np.median(arr)),
        "mean_r": float(np.mean(arr)),
        "min_r": float(np.min(arr)),
    }
