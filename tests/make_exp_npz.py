"""Pack the example data written by r_reference.R into tests/data/exp_rawdata.npz (counts < 2^16)."""
from pathlib import Path

import numpy as np
import scipy.io

D = Path(__file__).parent / "data"
X = scipy.io.mmread(D / "exp_counts.mtx").tocsc()
np.savez_compressed(D / "exp_rawdata.npz", data=X.data.astype(np.uint16), indices=X.indices.astype(np.uint16),
                    indptr=X.indptr.astype(np.int64), shape=np.array(X.shape),
                    genes=np.array((D / "exp_genes.txt").read_text().split("\n")[:-1]),
                    cells=np.array((D / "exp_cells.txt").read_text().split("\n")[:-1]))
for f in ("exp_counts.mtx", "exp_genes.txt", "exp_cells.txt"):
    (D / f).unlink()
