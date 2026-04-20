from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

IdType = Literal["Symbol", "Ensembl"]
Genome = Literal["hg20", "mm10"]
Distance = Literal["euclidean", "pearson", "spearman"]
Backend = Literal["cpu"]  # V1 only
BreakpointKS = Literal["mc", "analytic"]


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
    breakpoint_ks: BreakpointKS = "mc"
    """Breakpoint KS mode. ``"mc"`` (default) matches R copykat: draws
    ``mc=1000`` Poisson-Gamma posterior samples per window pair and calls
    :func:`scipy.stats.ks_2samp`. ``"analytic"`` (experimental) computes the
    KS statistic in closed form via :func:`scipy.stats.gamma.cdf` on a dense
    grid — no MC noise, ~5–20x faster, but the effective threshold differs:
    ``ks_cut`` is tuned against the MC noise floor (~0.04 at ``mc=1000``),
    so flipping to ``"analytic"`` may systematically lower KS values on
    noisy boundary pairs and is likely to require retuning ``ks_cut``.
    Default remains ``"mc"`` to preserve bit-identical behaviour vs. the
    R reference."""

    def __post_init__(self) -> None:
        if self.distance not in ("euclidean", "pearson", "spearman"):
            raise ValueError(
                f"distance must be euclidean/pearson/spearman, got {self.distance}"
            )
        if self.id_type not in ("Symbol", "Ensembl"):
            raise ValueError(f"id_type must be Symbol/Ensembl, got {self.id_type}")
        if self.genome not in ("hg20", "mm10"):
            raise ValueError(f"genome must be hg20/mm10, got {self.genome}")
        if self.breakpoint_ks not in ("mc", "analytic"):
            raise ValueError(
                f"breakpoint_ks must be mc/analytic, got {self.breakpoint_ks}"
            )
        self.output_dir = Path(self.output_dir)
