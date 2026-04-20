"""Slice all 3CA datasets and run R copykat on every patient.

Processes one dataset at a time; within each dataset, processes patients
serially. Resumable — if ``<sample>/r_out/<sample>_copykat_prediction.txt``
already exists, the R step is skipped for that patient. All stdout/stderr
is teed into a top-level ``run_all.log`` plus per-patient ``r_out/run.log``.

After ``run_all_benchmarks.py`` has produced R outputs for every patient,
invoke ``run_py_sweep.py`` to run pycopykat and regenerate the py↔R
comparison artifacts (the py↔R consistency story is the only benchmark
target; no external truth is used).

Layout after a full run::

    benchmarks/full/
      Gao2021_Breast/
        DCIS1/{counts.tsv, cells.csv, r_out/...}
        TNBC1/...
      Kim2020_Lung/
        P1028/...
        ...
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT / ".venv" / "bin" / "python"
SLICE_PY = PROJECT / "scripts" / "slice_dataset.py"
R_DRIVER = PROJECT / "scripts" / "run_r_copykat.R"
N_CORES = 8

# (dataset_folder, subdir_for_cells_file, patient_list, cancer_label)
DATASETS: list[tuple[str, str, list[str], str]] = [
    ("Data_Gao2021_Breast",       "Breast",
     ["DCIS1", "TNBC1", "TNBC2", "TNBC3"], "Gao2021_Breast"),
    ("Data_Kim2020_Lung",         "",
     ["P1028", "P0019", "P0034"], "Kim2020_Lung"),
    ("Data_Lee2020_Colorectal",   "",
     ["SMC16", "SMC09", "SMC21"], "Lee2020_Colorectal"),
    ("Data_Obradovic2021_Kidney", "",
     ["Patient4", "Patient5", "Patient2"], "Obradovic2021_Kidney"),
    ("Data_Qian2020_Ovarian",     "",
     ["11", "14", "12", "13"], "Qian2020_Ovarian"),
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


def _r_done(patient_dir: Path, sam_name: str) -> bool:
    return (patient_dir / "r_out" / f"{sam_name}_copykat_prediction.txt").exists()


def process_dataset(
    data_dir: str, subdir: str, patients: list[str], label: str, out_base: Path
) -> None:
    """Slice then run R copykat on each patient for one cancer."""
    ds_root = PROJECT / "tests" / "data" / data_dir
    slice_source = ds_root / subdir if subdir else ds_root
    cancer_out = out_base / label
    cancer_out.mkdir(parents=True, exist_ok=True)

    # Skip slicing for patients already fully done
    todo: list[str] = []
    for pt in patients:
        if _r_done(cancer_out / pt, f"{pt}_full"):
            _log(f"[{label}] skip {pt} — R output already present")
        else:
            todo.append(pt)

    if todo:
        rc = _run(
            [
                str(VENV_PY), str(SLICE_PY),
                "--dataset", str(slice_source),
                "--samples", ",".join(todo),
                "--out-base", str(cancer_out),
            ],
            log_path=cancer_out / "slice.log",
        )
        if rc != 0:
            _log(f"[{label}] !! slicing failed (rc={rc})")

    for pt in patients:
        pt_dir = cancer_out / pt
        sam_name = f"{pt}_full"
        r_out = pt_dir / "r_out"
        r_out.mkdir(parents=True, exist_ok=True)

        if _r_done(pt_dir, sam_name):
            _log(f"[{label}/{pt}] R prediction already present, skipping R step")
            continue

        counts_tsv = pt_dir / "counts.tsv"
        if not counts_tsv.exists():
            _log(f"[{label}/{pt}] !! missing counts.tsv, skip patient")
            continue
        _log(f"[{label}/{pt}] running R copykat…")
        t0 = time.time()
        rc = _run(
            [
                "Rscript", str(R_DRIVER),
                str(counts_tsv), str(r_out), sam_name, str(N_CORES),
            ],
            log_path=r_out / "run.log",
        )
        dt = time.time() - t0
        _log(f"[{label}/{pt}] R copykat rc={rc}, elapsed={dt/60:.2f} min")


def main() -> None:
    out_base = PROJECT / "benchmarks" / "full"
    out_base.mkdir(parents=True, exist_ok=True)
    _log("=== run_all_benchmarks starting ===")
    for ds, subdir, patients, label in DATASETS:
        _log(f"=== dataset: {label} ({len(patients)} patients) ===")
        process_dataset(ds, subdir, patients, label, out_base)
    _log("=== run_all_benchmarks finished ===")
    _log("Next step: run `scripts/run_py_sweep.py` to run pycopykat and "
         "regenerate the py↔R comparison artifacts.")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main() or 0)
