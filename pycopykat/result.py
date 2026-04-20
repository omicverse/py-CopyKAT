from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CopykatResult:
    """Output of pycopykat.copykat().

    cna_mat : bins x cells DataFrame with MultiIndex (chrom, start, end).
    prediction : cells x {cell, copykat.pred} DataFrame.
    linkage : scipy.cluster.hierarchy linkage matrix (n-1, 4).
    subclone : Series mapping aneuploid cell names -> subclone label (int).
    warnings : tuple of human-readable warning strings.
    """

    cna_mat: pd.DataFrame
    prediction: pd.DataFrame
    linkage: np.ndarray
    subclone: pd.Series
    warnings: tuple[str, ...]

    def to_txt(self, output_dir: Path, sam_name: str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.cna_mat.reset_index().to_csv(
            output_dir / f"{sam_name}_copykat_CNA_results.txt",
            sep="\t",
            index=False,
        )
        self.prediction.to_csv(
            output_dir / f"{sam_name}_copykat_prediction.txt",
            sep="\t",
            index=False,
        )

    def to_h5ad(self, path: Path) -> None:
        import anndata as ad

        # cells x bins
        X = self.cna_mat.T.to_numpy()  # noqa: N806  (anndata convention)
        var = self.cna_mat.index.to_frame(index=False)
        obs = self.prediction.set_index("cell")
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.write_h5ad(path)
