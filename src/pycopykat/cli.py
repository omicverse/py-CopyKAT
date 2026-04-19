"""``pycopykat`` CLI entry point (typer-based).

Installed as the ``pycopykat`` console script via ``pyproject.toml``::

    [project.scripts]
    pycopykat = "pycopykat.cli:app"

Invoke with ``pycopykat run --input counts.tsv --output-dir out/`` etc.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from pycopykat import CopykatConfig, copykat

app = typer.Typer(help="pycopykat — Python rewrite of CopyKAT")


@app.command()
def version() -> None:
    """Print the installed pycopykat version."""
    from pycopykat import __version__

    typer.echo(__version__)


@app.command()
def run(
    input: Path = typer.Option(..., "--input", help="genes×cells TSV/CSV/h5ad path"),
    output_dir: Path = typer.Option(Path("."), "--output-dir"),
    sam_name: str = typer.Option("sample", "--sam-name"),
    id_type: str = typer.Option("Symbol", "--id-type"),
    genome: str = typer.Option("hg20", "--genome"),
    cell_line: bool = typer.Option(False, "--cell-line/--no-cell-line"),
    ngene_chr: int = typer.Option(5, "--ngene-chr"),
    min_gene_per_cell: int = typer.Option(200, "--min-gene-per-cell"),
    low_dr: float = typer.Option(0.05, "--low-dr"),
    up_dr: float = typer.Option(0.1, "--up-dr"),
    win_size: int = typer.Option(25, "--win-size"),
    ks_cut: float = typer.Option(0.1, "--ks-cut"),
    distance: str = typer.Option("euclidean", "--distance"),
    n_jobs: int = typer.Option(1, "--n-jobs"),
    seed: int = typer.Option(1234, "--seed"),
    output_h5ad: bool = typer.Option(False, "--output-h5ad/--no-output-h5ad"),
) -> None:
    """Run the full copykat pipeline on a single counts file."""
    sep = "\t" if input.suffix in (".tsv", ".txt") else ","
    if input.suffix == ".h5ad":
        import anndata as ad  # type: ignore[import-untyped]

        a = ad.read_h5ad(input)
        X = a.X.toarray() if hasattr(a.X, "toarray") else a.X
        # AnnData convention: obs × var → transpose to genes × cells
        mat = pd.DataFrame(
            X.T, index=a.var_names.astype(str), columns=a.obs_names.astype(str)
        )
    else:
        mat = pd.read_csv(input, sep=sep, index_col=0)

    cfg = CopykatConfig(
        id_type=id_type,  # type: ignore[arg-type]
        genome=genome,  # type: ignore[arg-type]
        cell_line=cell_line,
        ngene_chr=ngene_chr,
        min_gene_per_cell=min_gene_per_cell,
        low_dr=low_dr,
        up_dr=up_dr,
        win_size=win_size,
        ks_cut=ks_cut,
        distance=distance,  # type: ignore[arg-type]
        n_jobs=n_jobs,
        seed=seed,
        sam_name=sam_name,
        output_dir=output_dir,
        output_h5ad=output_h5ad,
    )
    res = copykat(mat, config=cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    res.to_txt(output_dir, sam_name)
    if output_h5ad:
        res.to_h5ad(output_dir / f"{sam_name}_copykat.h5ad")
    typer.echo(f"wrote outputs under {output_dir}")


if __name__ == "__main__":
    app()
