from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

IdType = Literal["Symbol", "Ensembl"]
Genome = Literal["hg20"]  # V1: mm10 deferred
Distance = Literal["euclidean", "pearson", "spearman"]
Backend = Literal["cpu"]  # V1 only


@dataclass(slots=True)
class CopykatConfig:
    id_type: IdType = "Symbol"
    genome: Genome = "hg20"
    cell_line: bool = False
    ngene_chr: int = 5
    min_gene_per_cell: int = 200
    low_dr: float = 0.05
    up_dr: float = 0.1
    win_size: int = 25
    ks_cut: float = 0.1
    distance: Distance = "euclidean"
    norm_cell_names: list[str] | None = None
    sam_name: str = ""
    output_dir: Path = field(default_factory=lambda: Path("."))
    output_seg: bool = False
    output_h5ad: bool = False
    plot_genes: bool = True
    n_jobs: int = 1
    seed: int = 1234
    backend: Backend = "cpu"

    def __post_init__(self) -> None:
        if self.distance not in ("euclidean", "pearson", "spearman"):
            raise ValueError(
                f"distance must be euclidean/pearson/spearman, got {self.distance}"
            )
        if self.id_type not in ("Symbol", "Ensembl"):
            raise ValueError(f"id_type must be Symbol/Ensembl, got {self.id_type}")
        if self.genome != "hg20":
            raise ValueError(f"V1 only supports hg20; got {self.genome}")
        self.output_dir = Path(self.output_dir)
