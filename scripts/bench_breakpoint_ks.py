"""Microbenchmark: MC vs analytic Gamma-KS breakpoint detection on Kim P0019.

Runs the pycopykat pipeline on the Kim2020_Lung P0019 counts TSV up through
step 7 (producing ``norm_relat`` + cluster labels) by capturing the inputs
to ``segment_cells`` via a monkey-patched stub that short-circuits steps 8+.
Then, for each per-cluster consensus signal, times:

  * ``find_breakpoints`` (MC, ``mc=1000``, seed=1234)
  * ``find_breakpoints_analytic`` (closed-form Gamma-KS)

Each is repeated 3x; the median wall time is reported. A symmetric-diff
column counts breakpoints that appear in exactly one of the two sets.

Outputs:
  - stdout: formatted table
  - CSV:    benchmarks/scaling/b1_scaffold_microbench.csv

Usage:
  uv run python scripts/bench_breakpoint_ks.py
"""
from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

import pycopykat.segment.mcmc as _seg_mcmc
from pycopykat import CopykatConfig, copykat
from pycopykat.segment.breakpoint import (
    find_breakpoints,
    find_breakpoints_analytic,
)
from pycopykat.segment.mcmc import _consensus_per_cluster

REPO_ROOT = Path(__file__).resolve().parent.parent
COUNTS_TSV = REPO_ROOT / "benchmarks/full/Kim2020_Lung/P0019/counts.tsv"
OUT_CSV = REPO_ROOT / "benchmarks/scaling/b1_scaffold_microbench.csv"


class _SegmentInputsCaptured(Exception):
    """Raised inside the monkey-patched ``segment_cells`` to exit the
    pipeline early after the step-8 inputs have been captured."""

    def __init__(self, payload: dict) -> None:
        super().__init__("captured")
        self.payload = payload


def _capture_segment_inputs(mat: pd.DataFrame, cfg: CopykatConfig) -> dict:
    """Run the pipeline until ``segment_cells`` is invoked, then abort with
    the captured arguments.

    This avoids re-implementing the preprocessing + baseline machinery in the
    benchmark script (which would drift as the pipeline evolves).
    """
    original = _seg_mcmc.segment_cells

    def _stub(
        fttmat, clu, *, bins, ks_cut, seed, mc=1000, n_jobs=1,
        breakpoint_ks="mc",
    ):
        raise _SegmentInputsCaptured(
            {
                "fttmat": np.asarray(fttmat, dtype=np.float64).copy(),
                "clu": np.asarray(clu, dtype=np.int_).copy(),
                "bins": int(bins),
                "ks_cut": float(ks_cut),
                "seed": int(seed),
                "mc": int(mc),
            }
        )

    _seg_mcmc.segment_cells = _stub  # type: ignore[assignment]
    # Also patch the already-imported reference inside pipeline module.
    import pycopykat.pipeline as _pl

    _pl_orig = _pl.segment_cells
    _pl.segment_cells = _stub  # type: ignore[assignment]
    try:
        copykat(mat, config=cfg)
        raise RuntimeError("pipeline did not reach segment_cells")
    except _SegmentInputsCaptured as e:
        return e.payload
    finally:
        _seg_mcmc.segment_cells = original  # type: ignore[assignment]
        _pl.segment_cells = _pl_orig  # type: ignore[assignment]


@contextmanager
def _timed(store: list[float]) -> None:
    t0 = time.perf_counter()
    yield
    store.append(time.perf_counter() - t0)


def _time_mc(signal, bins, ks_cut, seed_base, reps=3):
    times: list[float] = []
    breaks_last: list[int] | None = None
    for _ in range(reps):
        with _timed(times):
            breaks_last = find_breakpoints(
                signal, bins=bins, ks_cut=ks_cut, seed=seed_base, mc=1000,
            )
    return median(times), breaks_last or []


def _time_analytic(signal, bins, ks_cut, reps=3):
    times: list[float] = []
    breaks_last: list[int] | None = None
    for _ in range(reps):
        with _timed(times):
            breaks_last = find_breakpoints_analytic(
                signal, bins=bins, ks_cut=ks_cut,
            )
    return median(times), breaks_last or []


def main() -> int:
    if not COUNTS_TSV.exists():
        raise SystemExit(f"counts TSV not found: {COUNTS_TSV}")

    print(f"[bench] reading counts: {COUNTS_TSV}")
    mat = pd.read_csv(COUNTS_TSV, sep="\t", index_col=0)
    print(f"[bench] mat: {mat.shape[0]} genes x {mat.shape[1]} cells")

    cfg = CopykatConfig(n_jobs=4, sam_name="P0019_microbench", seed=1234)
    print("[bench] driving pipeline through step 7 (capturing segment_cells args)...")
    payload = _capture_segment_inputs(mat, cfg)
    fttmat = payload["fttmat"]
    clu = payload["clu"]
    bins = payload["bins"]
    ks_cut = payload["ks_cut"]
    seed = payload["seed"]
    print(
        f"[bench] captured: fttmat shape {fttmat.shape}, clusters {sorted(set(clu.tolist()))}"
    )

    consensus = _consensus_per_cluster(fttmat, clu)
    n_consensus = consensus.shape[1]
    print(f"[bench] per-cluster consensus signals: {n_consensus}")

    rows: list[dict] = []
    for c in range(n_consensus):
        signal = consensus[:, c]
        mc_seed = seed + 1000 * c
        mc_time, mc_breaks = _time_mc(signal, bins, ks_cut, mc_seed)
        an_time, an_breaks = _time_analytic(signal, bins, ks_cut)
        ratio = mc_time / an_time if an_time > 0 else float("inf")
        mc_set = set(mc_breaks)
        an_set = set(an_breaks)
        sym_diff = len(mc_set.symmetric_difference(an_set))
        row = {
            "cluster_idx": c,
            "n_genes": int(signal.size),
            "mc_time_s": mc_time,
            "analytic_time_s": an_time,
            "speedup_ratio": ratio,
            "mc_breaks": len(mc_breaks),
            "analytic_breaks": len(an_breaks),
            "sym_diff": sym_diff,
        }
        rows.append(row)
        print(
            f"  cluster {c}: MC {mc_time:.3f}s  analytic {an_time:.3f}s  "
            f"ratio {ratio:.1f}x  MC_breaks={len(mc_breaks)} "
            f"an_breaks={len(an_breaks)} sym_diff={sym_diff}"
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[bench] wrote {OUT_CSV}")

    # Aggregate summary.
    tot_mc = sum(r["mc_time_s"] for r in rows)
    tot_an = sum(r["analytic_time_s"] for r in rows)
    print(
        f"\n[bench] TOTAL across {n_consensus} clusters: "
        f"MC {tot_mc:.3f}s  analytic {tot_an:.3f}s  "
        f"overall_ratio {tot_mc / tot_an if tot_an > 0 else float('inf'):.2f}x"
    )
    mc_break_total = sum(r["mc_breaks"] for r in rows)
    an_break_total = sum(r["analytic_breaks"] for r in rows)
    print(
        f"[bench] total breakpoints (sum over clusters): "
        f"MC {mc_break_total}  analytic {an_break_total}  "
        f"sym_diff_sum {sum(r['sym_diff'] for r in rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
