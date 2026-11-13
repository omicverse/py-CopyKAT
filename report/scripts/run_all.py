"""Benchmark driver: every task, twice, for R copykat and pycopykat (1 thread) on single idle cores
(taskset, one BLAS thread), and pycopykat with 8 threads on 8 idle cores. Results go to
<runs>/<engine>/<task>/run<k>/.

Usage: python run_all.py <tasks_dir> <runs_dir> <rscript> <python> [n_parallel] [repeats, e.g. 1,2] [engines]
"""
from __future__ import annotations

import os
import pathlib
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
tasks_dir, out_dir, rscript, python = map(str, sys.argv[1:5])
n_par = int(sys.argv[5]) if len(sys.argv) > 5 else 12
reps = [int(s) for s in (sys.argv[6] if len(sys.argv) > 6 else "1,2").split(",")]
engines = (sys.argv[7] if len(sys.argv) > 7 else "r,py1,py8").split(",")


def idle_cores(n):
    """The n allowed cores with the least load over one second (from /proc/stat)."""
    def snap():
        out = {}
        for line in open("/proc/stat"):
            if line.startswith("cpu") and line[3].isdigit():
                f = line.split()
                v = list(map(int, f[1:]))
                out[int(f[0][3:])] = (sum(v), v[3] + v[4])
        return out
    a = snap()
    time.sleep(1)
    b = snap()
    allowed = sorted(os.sched_getaffinity(0))
    busy = {c: 1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in allowed}
    return sorted(allowed, key=lambda c: (busy[c], c))[:n]


grab = threading.Lock()
cores = queue.Queue()
for c in idle_cores(n_par + 8):
    cores.put(c)
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           OMP_WAIT_POLICY="PASSIVE", PYTHONWARNINGS="ignore")
tasks = sorted((p for p in pathlib.Path(tasks_dir).iterdir() if (p / "counts.mtx").exists()),
               key=lambda p: -p.joinpath("counts.mtx").stat().st_size)          # largest first
jobs = []
for t in tasks:
    for r in reps:
        if "r" in engines:
            jobs.append(("r", [rscript, str(HERE / "02_r_copykat.R")], t, r, 1))
        if "py1" in engines:
            jobs.append(("py1", [python, str(HERE / "03_py_copykat.py")], t, r, 1))
        if "py8" in engines:
            jobs.append(("py8", [python, str(HERE / "03_py_copykat.py")], t, r, 8))


def run(job):
    engine, cmd, t, r, nthr = job
    od = pathlib.Path(out_dir) / engine / t.name / f"run{r}"
    if (od / "result.json").exists() or (od / "runinfo.tsv").exists():
        return
    with grab:                                   # take all the cores of a job at once
        got = [cores.get() for _ in range(nthr)]
    try:
        e = dict(env)
        if nthr > 1:
            e.update(OPENBLAS_NUM_THREADS=str(nthr), OMP_NUM_THREADS=str(nthr), MKL_NUM_THREADS=str(nthr))
        args = [str(t), str(od)] + ([str(nthr)] if engine != "r" else ["1"])
        res = subprocess.run(["taskset", "-c", ",".join(map(str, got)), *cmd, *args], env=e,
                             capture_output=True, text=True)
        print(engine, t.name, r, (res.stdout.strip().splitlines() or [""])[-1] or res.stderr.strip()[-400:], flush=True)
    finally:
        for c in got:
            cores.put(c)


with ThreadPoolExecutor(n_par) as ex:
    list(ex.map(run, jobs))
