from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

IdType = Literal["Symbol", "Ensembl"]
Genome = Literal["hg20", "mm10"]
Distance = Literal["euclidean", "pearson", "spearman"]


@dataclass(slots=True)
class CopykatConfig:
    """Arguments of R ``copykat()`` (same defaults), plus ``seed`` and ``n_jobs``.

    ``id_type`` is R's ``id.type`` ("Symbol" for "S", "Ensembl" for "E"); ``low_dr``/``up_dr`` are
    ``LOW.DR``/``UP.DR``; ``ks_cut`` is ``KS.cut``; ``norm_cell_names`` are known normal cells.
    ``seed`` is the value R copykat passes to ``set.seed`` (1234). ``n_jobs`` threads are used for
    the steps without random numbers; the results do not depend on it.
    """

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
    output_h5ad: bool = False
    n_jobs: int = 1
    seed: int = 1234

    def __post_init__(self) -> None:
        if self.distance not in ("euclidean", "pearson", "spearman"):
            raise ValueError(f"distance must be euclidean/pearson/spearman, got {self.distance}")
        if self.id_type not in ("Symbol", "Ensembl"):
            raise ValueError(f"id_type must be Symbol/Ensembl, got {self.id_type}")
        if self.genome not in ("hg20", "mm10"):
            raise ValueError(f"genome must be hg20/mm10, got {self.genome}")
        self.output_dir = Path(self.output_dir)
