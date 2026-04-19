"""Sweep pycopykat + compare-figure across every patient produced by run_all.

Assumes ``benchmarks/full/<cancer>/<patient>/`` already contains
``counts.tsv``, ``cells.csv`` and ``r_out/<sample>_copykat_prediction.txt``
from ``run_all_benchmarks.py``. For each patient, runs
``scripts/run_py_copykat.py`` then ``scripts/compare_py_vs_r.py`` and
aggregates a patient-level summary.

Resumable: skips patients whose ``figure_py_vs_R.png`` already exists.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT / ".venv" / "bin" / "python"
PY_RUNNER = PROJECT / "scripts" / "run_py_copykat.py"
COMPARE = PROJECT / "scripts" / "compare_py_vs_r.py"
N_JOBS = 8

# Must match run_all_benchmarks.py so sample names line up
DATASETS: list[tuple[str, list[str]]] = [
    ("Gao2021_Breast",       ["DCIS1", "TNBC1", "TNBC2", "TNBC3"]),
    ("Kim2020_Lung",         ["P1028", "P0019", "P0034"]),
    ("Lee2020_Colorectal",   ["SMC16", "SMC09", "SMC21"]),
    ("Obradovic2021_Kidney", ["Patient4", "Patient5", "Patient2"]),
    ("Qian2020_Ovarian",     ["11", "14", "12", "13"]),
]


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _run(cmd: list[str], log_path: Path | None = None) -> int:
    _log("$ " + " ".join(str(c) for c in cmd))
    if log_path is None:
        return subprocess.call(cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as lf:
        return subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)


def _r_ready(pt_dir: Path, sam_name: str) -> bool:
    return (pt_dir / "r_out" / f"{sam_name}_copykat_prediction.txt").exists()


def _py_figure_exists(pt_dir: Path) -> bool:
    return (pt_dir / "figure_py_vs_R.png").exists()


def _py_prediction_exists(pt_dir: Path, sam_name: str) -> bool:
    return (pt_dir / "py_out" / f"{sam_name}_py_copykat_prediction.txt").exists()


def process_patient(
    pt_dir: Path, sam_name: str, cancer: str, pt: str
) -> dict[str, object] | None:
    if not _r_ready(pt_dir, sam_name):
        _log(f"[{cancer}/{pt}] !! R prediction missing — skip")
        return None

    py_out = pt_dir / "py_out"
    py_out.mkdir(parents=True, exist_ok=True)

    if not _py_prediction_exists(pt_dir, sam_name):
        _log(f"[{cancer}/{pt}] running pycopykat…")
        t0 = time.time()
        rc = _run(
            [
                str(VENV_PY), str(PY_RUNNER),
                "--counts", str(pt_dir / "counts.tsv"),
                "--out", str(py_out),
                "--sam-name", sam_name,
                "--n-jobs", str(N_JOBS),
            ],
            log_path=py_out / "run.log",
        )
        dt = (time.time() - t0) / 60.0
        _log(f"[{cancer}/{pt}] pycopykat rc={rc} elapsed={dt:.2f} min")
        if rc != 0:
            _log(f"[{cancer}/{pt}] !! pycopykat failed")
            return None
    else:
        _log(f"[{cancer}/{pt}] py prediction exists — skip pycopykat step")

    # Always re-run compare so metrics.csv stays in the 3-row 'comparison'
    # format; run_all_benchmarks's benchmark_r_vs_truth.py step B overwrites
    # this file with a single-row R-vs-truth schema on retries.
    _log(f"[{cancer}/{pt}] building py-vs-R figure…")
    rc = _run(
        [
            str(VENV_PY), str(COMPARE),
            "--r-out", str(pt_dir / "r_out"),
            "--py-out", str(pt_dir / "py_out"),
            "--cells", str(pt_dir / "cells.csv"),
            "--sam-name", sam_name,
            "--out-dir", str(pt_dir),
            "--title-suffix", f"{cancer} {pt}",
        ],
        log_path=pt_dir / "compare.log",
    )
    if rc != 0:
        _log(f"[{cancer}/{pt}] !! compare failed")
        return None

    # aggregate metrics
    m_path = pt_dir / "metrics.csv"
    if not m_path.exists():
        return None
    df = pd.read_csv(m_path)
    row: dict[str, object] = {"cancer": cancer, "sample": pt}
    for _, r in df.iterrows():
        prefix = str(r["comparison"])
        for k in ("accuracy", "precision", "recall", "f1", "ari", "kappa", "fmi",
                  "n_cells"):
            if k in r and pd.notna(r[k]):
                row[f"{prefix}__{k}"] = r[k]
    return row


def main() -> None:
    base = PROJECT / "benchmarks" / "full"
    if not base.exists():
        print(f"!! {base} does not exist — did you run run_all_benchmarks.py?")
        return
    _log("=== pycopykat sweep starting ===")
    rows: list[dict[str, object]] = []
    for cancer, patients in DATASETS:
        _log(f"=== dataset: {cancer} ({len(patients)} patients) ===")
        cancer_dir = base / cancer
        for pt in patients:
            pt_dir = cancer_dir / pt
            if not pt_dir.exists():
                _log(f"[{cancer}/{pt}] !! dir missing — skip")
                continue
            sam_name = f"{pt}_full"
            row = process_patient(pt_dir, sam_name, cancer, pt)
            if row is not None:
                rows.append(row)
                pd.DataFrame(rows).to_csv(
                    base / "py_vs_r_summary.csv", index=False
                )
                _log(f"[checkpoint] wrote py_vs_r_summary.csv ({len(rows)} rows)")
    _log("=== pycopykat sweep finished ===")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main() or 0)
