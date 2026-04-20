"""Scaling benchmark: R copykat vs pycopykat at n_jobs = 1/4/8/16.

For each (sample, engine, n_jobs) triple, wraps the underlying runner with
``/usr/bin/time -v`` to capture wall-clock and peak resident-set size, then
aggregates into ``benchmarks/scaling/scaling_summary.csv``.

Layout::

    benchmarks/scaling/
      <sample>/
        py_n1/  {time.log, run.log, <sample>_py_copykat_prediction.txt, ...}
        py_n4/
        ...
        r_n1/   {time.log, run.log, <sample>_copykat_prediction.txt, ...}
        ...
      scaling_summary.csv

Runs are **strictly sequential** to keep timing clean (no CPU/RAM
contention). Resumable: if a configuration already has both
``time.log`` and a prediction file, it is skipped.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
VENV_PY = PROJECT / ".venv" / "bin" / "python"
PY_RUNNER = PROJECT / "scripts" / "run_py_copykat.py"
R_DRIVER = PROJECT / "scripts" / "run_r_copykat.R"

FULL = PROJECT / "benchmarks" / "full"
OUT_BASE = PROJECT / "benchmarks" / "scaling"

# (label, counts.tsv, sam_name)
SAMPLES: list[tuple[str, Path, str]] = [
    ("Kim2020_Lung_P0019",      FULL / "Kim2020_Lung" / "P0019" / "counts.tsv",      "P0019_full"),
    ("Qian2020_Ovarian_14",     FULL / "Qian2020_Ovarian" / "14" / "counts.tsv",     "14_full"),
]
N_JOBS_LIST: list[int] = [1, 4, 8, 16]
ENGINES: list[str] = ["py", "r"]

TIME_BIN = "/usr/bin/time"

# Regexes to parse /usr/bin/time -v output
_PATTERNS = {
    "wall_hms":        re.compile(r"Elapsed \(wall clock\) time.*:\s+(.+)$", re.M),
    "user_sec":        re.compile(r"User time \(seconds\):\s+([\d.]+)", re.M),
    "sys_sec":         re.compile(r"System time \(seconds\):\s+([\d.]+)", re.M),
    "cpu_pct":         re.compile(r"Percent of CPU this job got:\s+(\d+)%", re.M),
    "peak_rss_kb":     re.compile(r"Maximum resident set size \(kbytes\):\s+(\d+)", re.M),
    "page_faults":     re.compile(r"Major \(requiring I/O\) page faults:\s+(\d+)", re.M),
    "vol_ctx":         re.compile(r"Voluntary context switches:\s+(\d+)", re.M),
    "invol_ctx":       re.compile(r"Involuntary context switches:\s+(\d+)", re.M),
}


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _hms_to_sec(hms: str) -> float:
    parts = hms.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])


def _parse_time_log(path: Path) -> dict[str, object]:
    text = path.read_text()
    out: dict[str, object] = {}
    for key, rx in _PATTERNS.items():
        m = rx.search(text)
        if not m:
            continue
        out[key] = m.group(1)
    if "wall_hms" in out:
        out["wall_sec"] = _hms_to_sec(str(out["wall_hms"]))
    for k in ("user_sec", "sys_sec"):
        if k in out:
            out[k] = float(out[k])  # type: ignore[arg-type]
    for k in ("cpu_pct", "peak_rss_kb", "page_faults", "vol_ctx", "invol_ctx"):
        if k in out:
            out[k] = int(out[k])  # type: ignore[arg-type]
    return out


def _already_done(engine: str, out_dir: Path, sam_name: str) -> bool:
    time_log = out_dir / "time.log"
    if engine == "py":
        pred = out_dir / f"{sam_name}_py_copykat_prediction.txt"
    else:
        pred = out_dir / f"{sam_name}_copykat_prediction.txt"
    return time_log.exists() and pred.exists()


def _build_cmd(
    engine: str, counts: Path, out_dir: Path, sam_name: str, n_jobs: int
) -> list[str]:
    if engine == "py":
        return [
            str(VENV_PY), str(PY_RUNNER),
            "--counts", str(counts),
            "--out", str(out_dir),
            "--sam-name", sam_name,
            "--n-jobs", str(n_jobs),
        ]
    return [
        "Rscript", str(R_DRIVER),
        str(counts), str(out_dir), sam_name, str(n_jobs),
    ]


def run_one(
    engine: str, label: str, counts: Path, sam_name: str, n_jobs: int
) -> dict[str, object] | None:
    tag = f"{engine}_n{n_jobs}"
    out_dir = OUT_BASE / label / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    time_log = out_dir / "time.log"
    run_log = out_dir / "run.log"

    if _already_done(engine, out_dir, sam_name):
        _log(f"[{label}/{tag}] skip — already done")
    else:
        cmd = [TIME_BIN, "-v", "-o", str(time_log)] + _build_cmd(
            engine, counts, out_dir, sam_name, n_jobs
        )
        _log(f"[{label}/{tag}] $ {' '.join(cmd)}")
        t0 = time.time()
        with run_log.open("w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        _log(f"[{label}/{tag}] rc={rc} wall={dt/60:.2f} min")
        if rc != 0:
            _log(f"[{label}/{tag}] !! failed — continuing")
            return None

    if not time_log.exists():
        _log(f"[{label}/{tag}] !! time.log missing")
        return None

    row: dict[str, object] = {
        "sample": label, "engine": engine, "n_jobs": n_jobs,
    }
    row.update(_parse_time_log(time_log))

    # Also pull elapsed_min from the engine's own runinfo if present
    if engine == "py":
        rinfo = out_dir / f"{sam_name}_py_copykat_runinfo.txt"
    else:
        rinfo = out_dir / f"{sam_name}_copykat_runinfo.txt"
    if rinfo.exists():
        for line in rinfo.read_text().splitlines():
            if line.startswith("elapsed_min="):
                try:
                    row["script_elapsed_min"] = float(line.split("=", 1)[1])
                except ValueError:
                    pass
    return row


def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    if shutil.which(TIME_BIN) is None:
        _log(f"!! missing {TIME_BIN}")
        sys.exit(1)

    _log("=== scaling benchmark starting ===")
    rows: list[dict[str, object]] = []
    for label, counts, sam_name in SAMPLES:
        if not counts.exists():
            _log(f"!! counts missing: {counts}")
            continue
        _log(f"=== sample: {label} ({counts}) ===")
        for engine in ENGINES:
            for n in N_JOBS_LIST:
                row = run_one(engine, label, counts, sam_name, n)
                if row is not None:
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(
                        OUT_BASE / "scaling_summary.csv", index=False
                    )
                    _log(
                        f"[checkpoint] scaling_summary.csv now has {len(rows)} rows"
                    )
    _log("=== scaling benchmark finished ===")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main() or 0)
