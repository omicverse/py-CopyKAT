"""pycopykat on one task, mirroring 02_r_copykat.R.

Usage: python 03_py_copykat.py <task dir> <out dir> [n_jobs] [mode]
mode: auto (default), known (the published normal cells as norm_cell_names), cellline, pearson
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import scipy.io

import pycopykat
from pycopykat import core
from pycopykat.pipeline import _threads


def main():
    td, od = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    nj = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    mode = sys.argv[4] if len(sys.argv) > 4 else "auto"
    od.mkdir(parents=True, exist_ok=True)
    X = scipy.io.mmread(td / "counts.mtx").tocsc()
    genes = (td / "genes.txt").read_text().split("\n")[:-1]
    cells = (td / "cells.txt").read_text().split("\n")[:-1]
    pub = pd.read_csv(td / "published.txt", sep="\t", header=None, names=["cell", "pred"])
    kw = {}
    if mode == "known":
        kw["norm_cell_names"] = pub.cell[pub.pred == "N"].tolist()
    if mode == "cellline":
        kw["cell_line"] = True
    if mode == "pearson":
        kw["distance"] = "pearson"
    t0 = time.perf_counter()
    with _threads(nj):
        res = core.run(X, genes, cells, keep_intermediate=True, n_jobs=nj, **kw)
    el = time.perf_counter() - t0
    if res.prediction is not None:
        res.prediction.to_csv(od / "prediction.tsv", sep="\t", index=False)
    cna = res.cna.iloc[:, 3:]
    (od / "cna_cells.txt").write_text("\n".join(cna.columns) + "\n")
    cna.to_numpy().T.tofile(od / "cna.bin")                                 # R column-major order
    (od / "breaks.txt").write_text("\n".join(map(str, res.intermediate["breaks"])) + "\n")
    (od / "preN.txt").write_text("\n".join(res.preN) + "\n")
    (od / "result.json").write_text(json.dumps({
        "engine": "pycopykat", "version": pycopykat.__version__, "n_jobs": nj, "mode": mode,
        "seconds": el, "cells": int(cna.shape[1]), "WNS": res.WNS, "WNS1": res.WNS1}))
    print(td.name, mode, f"{cna.shape[1]} cells, {el:.1f} s, {nj} thread(s)", flush=True)


if __name__ == "__main__":
    main()
