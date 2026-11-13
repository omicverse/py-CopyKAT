"""Figure and the numbers quoted in the pycopykat technical report.

Usage: python 05_figures.py <bench_dir> <tasks_dir> <out_dir>
Compares R copykat 1.2.5 and pycopykat on the six breast tumours of the CopyKAT paper (GSE148673), both
run with copykat's defaults (set.seed(1234), n.cores = 1). <bench_dir> holds r/, py1/ and py8/ from
run_all.py.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
import scipy.io
import scipy.sparse as sp

BENCH, TASKS, OUT = map(pathlib.Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
C_R, C_PY, C_PY8 = "#eb6834", "#2a78d6", "#7fb2ea"
C_DIP, C_ANE = "#2f7f6f", "#b5a642"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})
TUMOURS = ["DCIS1", "TNBC1", "TNBC2", "TNBC3", "TNBC4", "TNBC5"]
EXAMPLE = "TNBC1"


def run_dir(engine, t, k=1):
    return BENCH / engine / t / f"run{k}"


def seconds(engine, t):
    out = []
    for k in (1, 2):
        d = run_dir(engine, t, k)
        if engine == "r":
            info = dict(l.split("\t", 1) for l in (d / "runinfo.tsv").read_text().strip().split("\n"))
            out.append(float(info["elapsed_s"]))
        else:
            out.append(json.load(open(d / "result.json"))["seconds"])
    return min(out)


def cna(engine, t):
    d = run_dir(engine, t)
    cells = (d / "cna_cells.txt").read_text().split()
    v = np.fromfile(d / "cna.bin")
    return pd.DataFrame(v.reshape(len(cells), -1).T, columns=cells)


def calls(engine, t):
    return pd.read_csv(run_dir(engine, t) / "prediction.tsv", sep="\t").set_index("cell.names")["copykat.pred"]


rows, keep = [], {}
for t in TUMOURS:
    rc, pc = cna("r", t), cna("py1", t)
    rp, pp = calls("r", t), calls("py1", t)
    rb = np.loadtxt(run_dir("r", t) / "breaks.txt", dtype=int)
    pb = np.loadtxt(run_dir("py1", t) / "breaks.txt", dtype=int)
    pub = pd.read_csv(TASKS / t / "published.txt", sep="\t", header=None, index_col=0)[1]
    called = pp[pp != "not.defined"]
    ane = called.str.contains("aneuploid")
    pubT = pub.reindex(called.index) == "T"
    py8 = calls("py8", t)
    row = dict(dataset=t, cells_in=int(len(pub)), cells=int(rc.shape[1]), bins=int(rc.shape[0]),
               same_cells=list(rc.columns) == list(pc.columns), breaks_R=int(rb.size), breaks_py=int(pb.size),
               same_breaks=bool(rb.size == pb.size and (rb == pb).all()),
               same_calls=bool(rp.equals(pp)), same_calls_8=bool(py8.equals(pp)),
               maxdiff=float(np.abs(rc.to_numpy() - pc.to_numpy()).max()),
               n_aneuploid=int(ane.sum()), n_diploid=int((~ane).sum()), n_pub_T=int((pub == "T").sum()),
               agree_pub=float((ane.to_numpy() == pubT.to_numpy()).mean()),
               sec_R=seconds("r", t), sec_py=seconds("py1", t), sec_py8=seconds("py8", t))
    res = json.load(open(run_dir("py1", t) / "result.json"))
    row["cor"] = res["classification"]["cor"]
    row["k_final"] = res["classification"]["k_final"]
    rows.append(row)
    keep[t] = (rc, pc, pp)
tab = pd.DataFrame(rows)
tab["speed"] = tab.sec_R / tab.sec_py
tab.to_csv(OUT / "comparison_per_dataset.tsv", sep="\t", index=False)
print(tab.to_string())

# ---- UMAP of the example (display only) ----
td = TASKS / EXAMPLE
X = sp.csr_matrix(scipy.io.mmread(td / "counts.mtx").T)
cells = (td / "cells.txt").read_text().split()
a = scanpy.AnnData(X.astype(np.float32))
scanpy.pp.normalize_total(a, target_sum=1e4)
scanpy.pp.log1p(a)
scanpy.pp.highly_variable_genes(a, n_top_genes=2000, subset=True)
scanpy.pp.pca(a, n_comps=30, random_state=0)
scanpy.pp.neighbors(a, n_neighbors=15, random_state=0)
scanpy.tl.umap(a, random_state=0)
xy = a.obsm["X_umap"]

# ======================= Figure 1 =======================
fig = plt.figure(figsize=(7.2, 4.7))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.42)


def panel(ax, s, dx=-0.36):
    ax.text(dx, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def crop(img):
    ink = (img[..., 3] > 0.04) & (img[..., :3].min(-1) < 0.96)
    ys, xs = np.where(ink)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


top = gs[0].subgridspec(1, 4, width_ratios=[0.62, 1, 1, 1], wspace=0.6)
logos = [crop(plt.imread(ASSETS / f)) for f in ["pycopykat_logo.png", "omicverse_logo.png"]]
lg = top[0].subgridspec(2, 1, hspace=0.25)
for k, img in enumerate(logos):
    ax = fig.add_subplot(lg[k])
    ax.imshow(img)
    ax.axis("off")
x = np.arange(len(tab))

axa = fig.add_subplot(top[1])
pts = []
for i, t in enumerate(TUMOURS):
    rv, pv = keep[t][0].to_numpy().ravel(), keep[t][1].to_numpy().ravel()
    pick = np.random.default_rng(i).choice(rv.size, min(4000, rv.size), replace=False)
    pts.append((rv[pick], pv[pick]))
rv = np.concatenate([q[0] for q in pts])
pv = np.concatenate([q[1] for q in pts])
lo, hi = min(rv.min(), pv.min()), max(rv.max(), pv.max())
axa.plot([lo, hi], [lo, hi], color=GRID, lw=0.6, zorder=0)
axa.scatter(rv, pv, s=0.6, color=C_PY, lw=0, rasterized=True)
axa.set_xlabel("copykat 1.2.5 (R)")
axa.set_ylabel("pycopykat")
axa.set_title("Copy-number matrix")
axa.set_aspect("equal")
axa.text(0.96, 0.04, f"6 tumours\nmax |R - Python|\n{tab.maxdiff.max():.0e}", transform=axa.transAxes,
         ha="right", va="bottom", fontsize=5, color=INK2)
panel(axa, "a")

axb = fig.add_subplot(top[2])
axb.bar(x, tab.agree_pub * 100, 0.6, color=C_PY)
for i, r in tab.iterrows():
    v = r.agree_pub * 100
    axb.text(i, v + 1, "100" if v == 100 else f"{v:.1f}", ha="center", va="bottom", fontsize=5, color=INK2)
axb.set_xticks(x, tab.dataset, rotation=40, ha="right", rotation_mode="anchor", fontsize=5.5)
axb.set_ylim(0, 112)
axb.set_ylabel("cells called as in the paper (%)")
axb.set_title("Calls as in the paper")
axb.grid(axis="y", color=GRID, lw=0.5)
axb.set_axisbelow(True)
panel(axb, "b")

axc = fig.add_subplot(top[3])
for col, c, lab in [("sec_R", C_R, "copykat 1.2.5 (R)"), ("sec_py", C_PY, "pycopykat, 1 thread"),
                    ("sec_py8", C_PY8, "pycopykat, 8 threads")]:
    axc.scatter(tab.cells, tab[col], s=12, color=c, edgecolor="white", linewidth=0.3, label=lab, zorder=3)
for _, r in tab.iterrows():
    axc.plot([r.cells] * 2, [r.sec_py, r.sec_R], color=GRID, lw=0.8, zorder=0)
axc.set_xscale("log")
axc.set_yscale("log")
axc.set_xticks([500, 1000, 3000], ["500", "1,000", "3,000"])
axc.set_yticks([30, 100, 300, 1000, 3000], ["30", "100", "300", "1,000", "3,000"])
axc.minorticks_off()
axc.set_xlabel("cells")
axc.set_ylabel("wall time (s)")
axc.set_title("Run time")
axc.grid(color=GRID, lw=0.5)
axc.set_axisbelow(True)
axc.legend(fontsize=5, loc="upper left", handletextpad=0.1)
panel(axc, "c")

# heatmaps of the example, cells grouped by the call, bins in genomic order
bot = gs[1].subgridspec(1, 2, width_ratios=[1.0, 0.36], wspace=0.12)
hm = bot[0].subgridspec(2, 1, hspace=0.1)
rc, pc, pp = keep[EXAMPLE]
chrom = pd.read_parquet(pathlib.Path(__import__("pycopykat").__file__).parent / "data" / "hg20_220kb_bins.parquet")
chrom = chrom.loc[chrom.chrom != 24, "chrom"].to_numpy()
bounds = np.flatnonzero(chrom[1:] != chrom[:-1]) + 1
order = sorted(rc.columns, key=lambda c: (pp[c] != "diploid", c))        # diploid cells first
n_dip = int((pp.reindex(order) == "diploid").sum())
for k, (nm, M) in enumerate([("R", rc), ("Python", pc)]):
    ax = fig.add_subplot(hm[k])
    ax.imshow(M[order].to_numpy().T, aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4, interpolation="nearest",
              rasterized=True)
    for b in bounds:
        ax.axvline(b - 0.5, color=INK2, lw=0.3)
    ax.axhline(n_dip - 0.5, color=INK, lw=0.6)
    ax.set_yticks([n_dip / 2, n_dip + (len(order) - n_dip) / 2], ["diploid", "aneuploid"], fontsize=5.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel(nm, fontsize=6)
    if k == 1:
        edges = np.concatenate(([0], bounds, [len(chrom)]))
        mids = (edges[:-1] + edges[1:]) / 2
        names = ["X" if c == 23 else ("" if c in (20, 22) else str(c)) for c in pd.unique(chrom)]
        ax.set_xticks(mids, names, fontsize=4.8)
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("chromosome (220 kb bins)")
    else:
        ax.set_xticks([])
        ax.set_title(f"{EXAMPLE}: copy-number profiles of {len(order):,} cells")
        panel(ax, "d", dx=-0.09)

ax = fig.add_subplot(bot[1])
lab = pp.reindex(cells).to_numpy()
for name, col in [("diploid", C_DIP), ("aneuploid", C_ANE)]:
    m = lab == name
    ax.scatter(xy[m, 0], xy[m, 1], s=0.8, c=col, lw=0, rasterized=True, label=f"{name} ({m.sum():,})")
ax.set_xticks([])
ax.set_yticks([])
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.set_aspect("equal")
ax.set_title(f"{EXAMPLE}: pycopykat calls")
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=5.5, handletextpad=0.1)
for hh in leg.legend_handles:
    hh.set_sizes([10])
panel(ax, "e", dx=-0.04)

fig.savefig(OUT / "fig1.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig1.png", bbox_inches="tight", dpi=300)

# ---- numbers quoted in the text ----
big = tab.loc[tab.cells.idxmax()]


def fmt(v):
    return f"{int(round(v)):,}"


nums = {
    "nTumours": len(tab), "nCellsTotal": fmt(tab.cells_in.sum()), "nCellsMin": fmt(tab.cells_in.min()),
    "nCellsMax": fmt(tab.cells_in.max()),
    "maxDiff": f"{tab.maxdiff.max():.0e}".replace("e-", "\\times 10^{-") + "}",
    "breaksMin": tab.breaks_R.min(), "breaksMax": tab.breaks_R.max(),
    "speedMin": f"{tab.speed.min():.1f}", "speedMax": f"{tab.speed.max():.0f}",
    "tBigR": fmt(big.sec_R), "tBigPy": fmt(big.sec_py), "tBigPyEight": fmt(big.sec_py8), "bigName": big.dataset,
    "agreePubMin": f"{tab.agree_pub[tab.dataset != 'TNBC4'].min() * 100:.1f}",
    "agreePubMax": f"{tab.agree_pub.max() * 100:.1f}",
    "agreeTNBCfour": f"{tab.agree_pub[tab.dataset == 'TNBC4'].iloc[0] * 100:.0f}",
    "corTNBCfour": f"{tab.cor[tab.dataset == 'TNBC4'].iloc[0]:.2f}",
    "nCalledTNBCfour": fmt(tab.cells[tab.dataset == "TNBC4"].iloc[0]),
    "nPubTTNBCfour": fmt(tab.n_pub_T[tab.dataset == "TNBC4"].iloc[0]),
}
(OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in nums.items()))
print(nums)
