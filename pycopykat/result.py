from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pycopykat._rcompat.hclust import HClust


@dataclass
class CopykatResult:
    """Output of :func:`pycopykat.copykat`, as R copykat's list.

    prediction : DataFrame with R's columns ``cell.names`` and ``copykat.pred`` ("diploid",
        "aneuploid", "c1:diploid:low.conf", "c2:aneuploid:low.conf" or "not.defined"); empty in
        cell-line mode, where R returns no prediction.
    cna_mat : copy-number matrix, bins x cells (hg20; index ``chrom, chrompos, abspos``) or
        genes x cells (mm10; index the seven annotation columns), as R's ``CNAmat``.
    hclustering : R's ``hclust`` of the cells on the final matrix (``merge``, ``height``,
        ``order``).
    warnings : R's two quality messages (``WNS1``, ``WNS``) when not empty.
    preN : the cells taken as diploid to set the baseline.
    """

    prediction: pd.DataFrame
    cna_mat: pd.DataFrame
    hclustering: HClust
    warnings: tuple[str, ...] = ()
    preN: list[str] = field(default_factory=list)

    def to_txt(self, output_dir: Path, sam_name: str) -> None:
        """Write ``<sam_name>_copykat_prediction.txt`` and ``<sam_name>_copykat_CNA_results.txt``
        (tab-separated, as R)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.cna_mat.reset_index().to_csv(output_dir / f"{sam_name}_copykat_CNA_results.txt",
                                          sep="\t", index=False)
        self.prediction.to_csv(output_dir / f"{sam_name}_copykat_prediction.txt", sep="\t", index=False)

    def to_h5ad(self, path: Path) -> None:
        import anndata as ad

        X = self.cna_mat.T.to_numpy()  # noqa: N806  (cells x bins)
        var = self.cna_mat.index.to_frame(index=False)
        var.index = var.index.astype(str)
        obs = self.prediction.set_index("cell.names").reindex(self.cna_mat.columns)
        adata = ad.AnnData(X=np.asarray(X, dtype=np.float64), obs=obs, var=var)
        adata.write_h5ad(path)
