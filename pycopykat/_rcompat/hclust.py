"""R's ``hclust`` (``ward.D`` and ``ward.D2``) and ``cutree``.

``hclust`` follows R's Fortran routine (F. Murtagh's nearest-neighbour list, ``hclust.f``): at
every step the pair with the smallest nearest-neighbour dissimilarity is merged (the first one on
ties), the dissimilarities to the new cluster are updated with the Lance-Williams formula in the
same order of operations, and the nearest neighbours are refreshed as R does, including the fix
for neighbours to the left of the new cluster. ``ward.D2`` squares the dissimilarities first and
reports square roots as heights. ``cutree(k)`` undoes the last ``k - 1`` merges and numbers the
groups in the order in which the observations first appear, as R does.

The update of each step is vectorised over the remaining clusters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_INF = 1e300


@dataclass
class HClust:
    """An R ``hclust`` result: ``merge`` (n-1 x 2, R's signed convention), ``height`` and
    ``order`` (1-based, as in R), plus the raw merge pairs (0-based)."""

    merge: np.ndarray
    height: np.ndarray
    order: np.ndarray
    pairs: np.ndarray
    labels: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.pairs.shape[0] + 1

    def cutree(self, k: int) -> np.ndarray:
        return cutree(self.pairs, k)


def hclust(d: np.ndarray, method: str = "ward.D", labels=None) -> HClust:
    """``hclust(as.dist(d), method)`` for a square symmetric dissimilarity matrix ``d``."""
    if method not in ("ward.D", "ward.D2"):
        raise ValueError("method must be 'ward.D' or 'ward.D2'")
    D = np.array(d, dtype=np.float64, copy=True)
    n = D.shape[0]
    if n < 2:
        raise ValueError("must have n >= 2 objects to cluster")
    if method == "ward.D2":
        D = D * D
    np.fill_diagonal(D, _INF)
    idx = np.arange(n)
    membr = np.ones(n)
    active = np.ones(n, dtype=bool)
    # nearest neighbour to the right of each i (first minimum)
    upper = np.where(idx[None, :] > idx[:, None], D, np.inf)
    nn = np.argmin(upper, axis=1)
    disnn = upper[idx, nn]
    del upper
    nn[n - 1] = -1
    disnn[n - 1] = np.inf
    pairs = np.empty((n - 1, 2), dtype=np.int64)
    crit = np.empty(n - 1)
    for step in range(n - 1):
        cand = np.where(active[: n - 1], disnn[: n - 1], np.inf)
        im = int(np.argmin(cand))
        dmin = cand[im]
        jm = int(nn[im])
        i2, j2 = min(im, jm), max(im, jm)
        pairs[step] = (i2, j2)
        crit[step] = np.sqrt(dmin) if method == "ward.D2" else dmin
        active[j2] = False
        ks = np.flatnonzero(active)
        ks = ks[ks != i2]
        if ks.size:
            d12 = D[i2, j2]
            mk = membr[ks]
            new = ((membr[i2] + mk) * D[i2, ks] + (membr[j2] + mk) * D[j2, ks]) - mk * d12
            new = new / ((membr[i2] + membr[j2]) + mk)
            D[i2, ks] = new
            D[ks, i2] = new
            right = ks > i2
            if right.any():
                r = int(np.argmin(new[right]))
                dnew, jj = new[right][r], int(ks[right][r])
            else:
                dnew, jj = np.inf, -1
            left = ~right
            if left.any():
                kl, vl = ks[left], new[left]
                closer = vl < disnn[kl]
                disnn[kl[closer]] = vl[closer]
                nn[kl[closer]] = i2
        else:
            dnew, jj = np.inf, -1
        membr[i2] += membr[j2]
        disnn[i2] = dnew
        nn[i2] = jj
        rows = np.flatnonzero(active[: n - 1] & ((nn[: n - 1] == i2) | (nn[: n - 1] == j2)))
        if rows.size:
            sub = D[rows]
            ok = active[None, :] & (idx[None, :] > rows[:, None])
            sub = np.where(ok, sub, np.inf)
            jmin = np.argmin(sub, axis=1)
            dm = sub[np.arange(rows.size), jmin]
            none = ~np.isfinite(dm)
            nn[rows] = np.where(none, -1, jmin)
            disnn[rows] = np.where(none, np.inf, dm)
    merge, order = _hcass2(pairs + 1, n)
    return HClust(merge=merge, height=crit, order=order, pairs=pairs,
                  labels=None if labels is None else np.asarray(labels))


def _hcass2(pairs1: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """R's ``hcass2``: the signed merge matrix and the leaf order (both 1-based)."""
    ia = pairs1[:, 0].tolist()
    ib = pairs1[:, 1].tolist()
    iia, iib = list(ia), list(ib)
    last = {}                      # cluster representative -> step that last merged it
    for i in range(n - 1):
        a, b = ia[i], ib[i]
        iia[i] = last.get(a, -a)
        iib[i] = last.get(b, -b)
        last[min(a, b)] = i + 1
    for i in range(n - 1):
        if iia[i] > 0 and iib[i] < 0:
            iia[i], iib[i] = iib[i], iia[i]
        if iia[i] > 0 and iib[i] > 0:
            iia[i], iib[i] = min(iia[i], iib[i]), max(iia[i], iib[i])
    order = [iia[n - 2], iib[n - 2]]
    for i in range(n - 2, 0, -1):
        j = order.index(i)
        order[j:j + 1] = [iia[i - 1], iib[i - 1]]
    return np.column_stack([iia, iib]).astype(np.int64), -np.asarray(order, dtype=np.int64)


def cutree(pairs: np.ndarray, k: int) -> np.ndarray:
    """``cutree(tree, k)`` from the 0-based merge pairs: 1-based group numbers in the order of
    first appearance."""
    n = pairs.shape[0] + 1
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}")
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs[: n - k]:
        ra, rb = find(int(a)), find(int(b))
        parent[rb] = ra
    roots = np.array([find(i) for i in range(n)])
    _, first, inv = np.unique(roots, return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first)] = np.arange(1, first.size + 1)
    return rank[inv]
