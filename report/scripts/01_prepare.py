"""Prepare the benchmark tasks: the six GSE148673 breast tumours of the CopyKAT paper (Gao et al. 2021).

Usage: python 01_prepare.py <raw dir with GSM*.txt.gz> <task dir>
Each task holds the UMI counts (genes x cells, Matrix Market), gene and cell names, and the
published CopyKAT call of each cell (N = diploid/normal, T = aneuploid/tumour)."""
import gzip, sys, pathlib
import numpy as np, pandas as pd, scipy.sparse as sp, scipy.io

raw, out_root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
for f in sorted(raw.glob("GSM*.txt.gz")):
    s = f.name.split("_")[-1].split(".")[0]
    with gzip.open(f, "rt") as fh:
        cells = fh.readline().rstrip("\n").split("\t")
        pred = fh.readline().rstrip("\n").split("\t")[1:]
        fh.readline()
        df = pd.read_csv(fh, sep="\t", header=None, index_col=0)
    if len(cells) == df.shape[1] + 1:
        cells = cells[1:]
    df = df[~df.index.duplicated()]
    out = out_root / s
    out.mkdir(parents=True, exist_ok=True)
    scipy.io.mmwrite(str(out / "counts.mtx"), sp.csc_matrix(df.to_numpy(dtype=np.float64)))
    pd.Series(df.index).to_csv(out / "genes.txt", index=False, header=False)
    pd.Series(cells).to_csv(out / "cells.txt", index=False, header=False)
    pd.Series(pred, index=cells).to_csv(out / "published.txt", sep="\t", header=False)
    print(s, df.shape, pd.Series(pred).value_counts().to_dict())
