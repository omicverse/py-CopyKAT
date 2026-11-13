"""R's ``rgamma`` on R's random stream, for MCMCpack's ``MCpoissongamma``.

CopyKAT's segmentation calls ``MCpoissongamma(y, alpha, 1, mc = 1000)`` once per window and per
cell; that is ``rgamma(1000, alpha + sum(y), 1 + length(y))``. R draws a gamma variate with
Ahrens & Dieter's GD algorithm (``a >= 1``) or GS (``a < 1``), both rejection samplers that use a
variable number of uniforms: a GD draw takes a normal (two uniforms, by inversion), then a uniform
for the squeeze and quotient tests, then a loop of exponential + uniform draws in the rare
rejections. Every draw starts where the previous one stopped, so the draws of a call depend on
all the calls before it.

:func:`rgamma_scalar` is a line-by-line port of R's ``rgamma.c`` and ``sexp.c``.
:class:`GammaStream` produces the same draws in vectorised form: for a batch of calls it evaluates
the length of a draw starting at every position of a window of the stream (the length depends
only on the uniforms from that position on and on the shape), walks the chain of draws of every
call at once from a guessed start, and then joins the calls in order: the true chain of a call
meets its guessed chain after a few draws, because two decodings of the same stream fall onto
the same position quickly. Calls whose start falls outside their window are decoded again from
their true start.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtri

from .rbase import qnorm
from .rng import RRandom

SQRT32 = 5.656854
EXP_M1 = 0.36787944117144232159
Q1, Q2, Q3, Q4, Q5, Q6, Q7 = 0.04166669, 0.02083148, 0.00801191, 0.00144121, -7.388e-5, 2.4511e-4, 2.424e-4
A1, A2, A3, A4, A5, A6, A7 = 0.3333333, -0.250003, 0.2000062, -0.1662921, 0.1423657, -0.1367177, 0.1233795
TAU = -0.71874483771719
EXPQ = (0.6931471805599453, 0.9333736875190459, 0.9888777961838675, 0.9984959252914960,
        0.9998292811061389, 0.9999833164100727, 0.9999985691438767, 0.9999998906925558,
        0.9999999924734159, 0.9999999995283275, 0.9999999999728814, 0.9999999999985598,
        0.9999999999999289, 0.9999999999999968, 0.9999999999999999, 1.0000000000000000)
BIG = 134217728.0
_I2_32M1 = 2.328306437080797e-10


# ---------------------------------------------------------------------------------------------
# scalar reference


def _qnorm1(p: float) -> float:
    return float(qnorm(np.array([p]), log=lambda x: np.array([math.log(float(x[0]))]))[0])


def rgamma_scalar(unif, a: float, scale: float) -> float:
    """One ``rgamma(1, a, scale = scale)`` draw; ``unif()`` returns R's next ``unif_rand()``."""

    def exp_rand() -> float:
        acc = 0.0
        u = unif()
        while u <= 0.0 or u >= 1.0:
            u = unif()
        while True:
            u += u
            if u > 1.0:
                break
            acc += EXPQ[0]
        u -= 1.0
        if u <= EXPQ[0]:
            return acc + u
        i = 0
        ustar = unif()
        umin = ustar
        while True:
            ustar = unif()
            if umin > ustar:
                umin = ustar
            i += 1
            if not (u > EXPQ[i]):
                break
        return acc + umin * EXPQ[0]

    if a < 1.0:
        if a == 0.0:
            return 0.0
        e = 1.0 + EXP_M1 * a
        while True:
            p = e * unif()
            if p >= 1.0:
                x = -math.log((e - p) / a)
                if exp_rand() >= (1.0 - a) * math.log(x):
                    break
            else:
                x = math.exp(math.log(p) / a)
                if exp_rand() >= x:
                    break
        return scale * x
    s2 = a - 0.5
    s = math.sqrt(s2)
    d = SQRT32 - s * 12.0
    u1 = unif()
    u1 = math.floor(BIG * u1) + unif()
    t = _qnorm1(u1 / BIG)
    x = s + 0.5 * t
    ret = x * x
    if t >= 0.0:
        return scale * ret
    u = unif()
    if d * u <= t * t * t:
        return scale * ret
    q0, b, si, c = _step4(a, s, s2)
    if x > 0.0:
        v = t / (s + s)
        if abs(v) <= 0.25:
            q = q0 + 0.5 * t * t * ((((((A7 * v + A6) * v + A5) * v + A4) * v + A3) * v + A2) * v + A1) * v
        else:
            q = q0 - s * t + 0.25 * t * t + (s2 + s2) * math.log(1.0 + v)
        if math.log(1.0 - u) <= q:
            return scale * ret
    while True:
        e = exp_rand()
        u = unif()
        u = u + u - 1.0
        t = b - si * e if u < 0.0 else b + si * e
        if t >= TAU:
            v = t / (s + s)
            if abs(v) <= 0.25:
                q = q0 + 0.5 * t * t * ((((((A7 * v + A6) * v + A5) * v + A4) * v + A3) * v + A2) * v + A1) * v
            else:
                q = q0 - s * t + 0.25 * t * t + (s2 + s2) * math.log(1.0 + v)
            if q > 0.0:
                w = math.expm1(q)
                if c * abs(u) <= w * math.exp(e - 0.5 * t * t):
                    break
    x = s + 0.5 * t
    return scale * x * x


def _step4(a, s, s2):
    """GD step 4 constants (scalars or arrays)."""
    r = 1.0 / a
    q0 = ((((((Q7 * r + Q6) * r + Q5) * r + Q4) * r + Q3) * r + Q2) * r + Q1) * r
    lo, mid = a <= 3.686, a <= 13.022
    b = np.where(lo, 0.463 + s + 0.178 * s2, np.where(mid, 1.654 + 0.0076 * s2, 1.77))
    si = np.where(lo, 1.235, np.where(mid, 1.68 / s + 0.275, 0.75))
    c = np.where(lo, 0.195 / s - 0.079 + 0.16 * s, np.where(mid, 0.062 / s + 0.024, 0.1515 / s))
    if np.ndim(a) == 0:
        return float(q0), float(b), float(si), float(c)
    return q0, b, si, c


# ---------------------------------------------------------------------------------------------
# random access to the stream


class _Uniforms:
    """R's ``unif_rand()`` values from the current position of an :class:`RRandom`, addressed by
    their offset, in one buffer that grows at the end and drops what has been used; the
    generator states at chunk starts are kept so the generator can be left exactly after the last
    value used."""

    CHUNK = 1 << 21

    def __init__(self, rng: RRandom):
        self.bits = rng._bits
        self.buf = np.empty(0)
        self.buf_start = 0                 # offset of buf[0]
        self.gen_end = 0                   # offset after the last generated value
        self.states: list[tuple[int, dict]] = []
        self.keep = 0                      # values before this offset may be dropped

    def _generate(self) -> np.ndarray:
        self.states.append((self.gen_end, self.bits.state))
        raw = self.bits.random_raw(self.CHUNK)
        y = raw * 2.3283064365386963e-10
        if not raw.all():                                  # R's fixup() keeps values in (0, 1)
            y[raw == 0] = 0.5 * _I2_32M1
        self.gen_end += self.CHUNK
        return y

    def view(self, lo: int, hi: int) -> tuple[np.ndarray, int]:
        """An array covering offsets ``[lo, hi)`` and the offset of its first element."""
        if lo < self.buf_start:
            raise RuntimeError("stream position already released")
        if hi > self.gen_end:
            new = [self._generate() for _ in range(-(-(hi - self.gen_end) // self.CHUNK))]
            self.buf = np.concatenate([self.buf[self.keep - self.buf_start:], *new])
            self.buf_start = self.keep
        return self.buf, self.buf_start

    def release(self, pos: int) -> None:
        """Forget the values before offset ``pos`` (dropped when the buffer is next extended)."""
        self.keep = pos
        while len(self.states) > 1 and self.states[1][0] <= pos:
            self.states.pop(0)

    def finish(self, pos: int) -> None:
        """Leave the generator just after offset ``pos - 1``."""
        for off, st in reversed(self.states):
            if off <= pos:
                self.bits.state = st
                self.bits.random_raw(pos - off)
                return
        self.bits.random_raw(pos - self.gen_end)


# ---------------------------------------------------------------------------------------------
# vectorised draws


def _q_quotient(t, v, s, s2, q0, log=np.log):
    """``q`` of GD steps 6 and 10."""
    poly = ((((((A7 * v + A6) * v + A5) * v + A4) * v + A3) * v + A2) * v + A1) * v
    small = q0 + 0.5 * t * t * poly
    with np.errstate(invalid="ignore", divide="ignore"):
        large = q0 - s * t + 0.25 * t * t + (s2 + s2) * log(1.0 + v)
    return np.where(np.abs(v) <= 0.25, small, large)


def _exp_rand(U: np.ndarray, r: np.ndarray):
    """``exp_rand()`` starting at offsets ``r`` of ``U``: (uniforms used, value)."""
    u = U[r].copy()
    acc = np.zeros(r.size)
    act = np.ones(r.size, dtype=bool)
    while act.any():
        u[act] += u[act]
        brk = act & (u > 1.0)
        grow = act & ~brk
        acc[grow] += EXPQ[0]
        act = grow
    u -= 1.0
    used = np.ones(r.size, dtype=np.int64)
    val = acc + u
    big = np.flatnonzero(u > EXPQ[0])
    if big.size:
        ub = u[big]
        pos = r[big] + 1
        umin = U[pos].copy()
        pos = pos + 1
        i = np.zeros(big.size, dtype=np.int64)
        act = np.ones(big.size, dtype=bool)
        while act.any():
            ustar = U[pos[act]]
            umin[act] = np.minimum(umin[act], ustar)
            pos[act] += 1
            i[act] += 1
            act[act] = ub[act] > np.asarray(EXPQ)[i[act]]
        used[big] = pos - r[big]
        val[big] = acc[big] + umin * EXPQ[0]
    return used, val


def _draw_lengths(U: np.ndarray, p: np.ndarray, s, s2, d, q0, b, si, c):
    """Uniforms used by a GD draw starting at each offset ``p`` of ``U`` (shape constants are
    per-offset arrays), and ``t`` of the draws accepted in steps 8-11 (NaN elsewhere)."""
    n = p.size
    used = np.full(n, 2, dtype=np.int64)
    t8 = np.full(n, np.nan)
    pp = (np.floor(BIG * U[p]) + U[p + 1]) / BIG
    neg = np.flatnonzero(pp < 0.5)
    if neg.size == 0:
        return used, t8
    used[neg] = 3
    t = ndtri(pp[neg])
    u = U[p[neg] + 2]
    lhs, rhs = d[neg] * u, t * t * t
    # decisions within a hair of the boundary are taken with R's qnorm instead of ndtri
    near = np.abs(lhs - rhs) <= 1e-9 * (np.abs(lhs) + np.abs(rhs))
    if near.any():
        t[near] = qnorm(pp[neg][near])
        rhs[near] = t[near] * t[near] * t[near]
    rej = lhs > rhs
    idx = np.flatnonzero(rej)
    if idx.size == 0:
        return used, t8
    g = neg[idx]
    tg, ug = t[idx], u[idx]
    sg, s2g, q0g = s[g], s2[g], q0[g]
    x = sg + 0.5 * tg
    with np.errstate(invalid="ignore", divide="ignore"):
        q = _q_quotient(tg, tg / (sg + sg), sg, s2g, q0g)
        lu = np.log(1.0 - ug)
    near = (x > 0.0) & (np.abs(lu - q) <= 1e-9 * (np.abs(lu) + np.abs(q)))
    if near.any():
        te = qnorm(pp[g[near]])
        q[near] = _q_quotient(te, te / (sg[near] + sg[near]), sg[near], s2g[near], q0g[near])
    acc7 = (x > 0.0) & (lu <= q)
    loop = g[~acc7]
    # steps 8-11, repeated until accepted
    r = p[loop] + 3
    todo = np.arange(loop.size)
    while todo.size:
        k = loop[todo]
        e_used, e = _exp_rand(U, r[todo])
        uu = U[r[todo] + e_used]
        uu = uu + uu - 1.0
        tt = np.where(uu < 0.0, b[k] - si[k] * e, b[k] + si[k] * e)
        ok = tt >= TAU
        with np.errstate(invalid="ignore", over="ignore"):
            q = _q_quotient(tt, tt / (s[k] + s[k]), s[k], s2[k], q0[k])
            ok &= (q > 0.0) & (c[k] * np.abs(uu) <= np.expm1(q) * np.exp(e - 0.5 * tt * tt))
        r[todo] += e_used + 1
        done = todo[ok]
        used[loop[done]] = r[done] - p[loop[done]]
        t8[loop[done]] = tt[ok]
        todo = todo[~ok]
    return used, t8


def _gd_constants(a):
    a = np.asarray(a, dtype=np.float64)
    s2 = a - 0.5
    s = np.sqrt(s2)
    d = SQRT32 - s * 12.0
    q0, b, si, c = _step4(a, s, s2)
    return s, s2, d, q0, b, si, c


# (E[uniforms per draw] - 2.5) * sqrt(a) on a log grid of a from 1 to 1e6 (8e6 positions each);
# used only to place the windows, so its precision affects speed, not the draws.
_EL_GRID = np.exp(np.linspace(0.0, np.log(1e6), 49))
_EL_VAL = np.array([0.8208, 0.85806, 0.96617, 1.14891, 1.41888, 0.28212, 0.26468, 0.25249, 0.24405,
                    0.24611, 0.24476, 0.244, 0.24327, 0.24292, 0.24147, 0.24062, 0.2403, 0.24,
                    0.23976, 0.2396, 0.23924, 0.23877, 0.23913, 0.23752, 0.2379, 0.23704, 0.23549,
                    0.23527, 0.23552, 0.23456, 0.23329, 0.23127, 0.22543, 0.22232, 0.22026, 0.2142,
                    0.21213, 0.20825, 0.2008, 0.19617, 0.19069, 0.18195, 0.17263, 0.16313, 0.14494,
                    0.13264, 0.1107, 0.09634, 0.07263])


def _expected_length(a: np.ndarray) -> np.ndarray:
    return 2.5 + np.interp(np.log(a), np.log(_EL_GRID), _EL_VAL) / np.sqrt(a)


class GammaStream:
    """Sequential ``rgamma(mc, shape[k], rate[k])`` calls on R's stream.

    ``calls(shapes, rates, mc)`` returns the ``mean`` of each call's draws (R's ``mean``, long
    double with the correction pass) or, with ``draws=True``, the draws themselves, and advances
    ``rng`` exactly as R does. ``n_jobs`` threads share the work that does not depend on the order
    of the calls."""

    PAD = 256

    def __init__(self, rng: RRandom, batch: int = 128, n_jobs: int = 1):
        self.rng = rng
        self.batch = batch
        self.n_jobs = max(1, int(n_jobs))
        self._pool = None

    def _map(self, fn, parts):
        if self.n_jobs == 1 or len(parts) == 1:
            return [fn(*p) for p in parts]
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(self.n_jobs)
        return list(self._pool.map(lambda p: fn(*p), parts))

    def calls(self, shapes, rates, mc: int, draws: bool = False) -> np.ndarray:
        shapes = np.asarray(shapes, dtype=np.float64)
        scales = 1.0 / np.asarray(rates, dtype=np.float64)
        n = shapes.size
        out = np.empty((n, mc)) if draws else np.empty(n)
        U = _Uniforms(self.rng)
        pos = 0
        k0 = 0
        gs = shapes < 1.0
        try:
            while k0 < n:
                if gs[k0]:                                   # GS calls (shape < 1): one by one
                    vals, pos = self._scalar_call(U, pos, shapes[k0], scales[k0], mc)
                    k1 = k0 + 1
                    vals = vals[None, :]
                else:
                    k1 = min(n, k0 + self.batch)
                    nxt = np.flatnonzero(gs[k0:k1])
                    if nxt.size:
                        k1 = k0 + int(nxt[0])
                    got = self._batch_region(U, pos, shapes[k0:k1], scales[k0:k1], mc)
                    if got is None:
                        got = self._batch(U, pos, shapes[k0:k1], scales[k0:k1], mc)
                    vals, pos = got
                out[k0:k1] = vals if draws else _r_row_means(vals)
                U.release(pos)
                k0 = k1
        finally:
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None
        U.finish(pos)
        return out

    @staticmethod
    def _scalar_call(U, pos, a, scale, mc):
        cur = [pos]

        def unif():
            buf, b0 = U.view(cur[0], cur[0] + 1)
            v = buf[cur[0] - b0]
            cur[0] += 1
            return float(v)

        vals = np.array([rgamma_scalar(unif, float(a), float(scale)) for _ in range(mc)])
        return vals, cur[0]

    def _lengths(self, buf, p, cst_rows):
        """``_draw_lengths`` over offsets ``p`` with per-offset constants, split over threads."""
        parts = np.array_split(np.arange(p.size), self.n_jobs) if p.size > 4096 else [np.arange(p.size)]
        res = self._map(lambda ix: _draw_lengths(buf, p[ix], *[c[ix] for c in cst_rows]),
                        [(ix,) for ix in parts])
        return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])

    def _batch(self, U, pos, a, scale, mc):
        K = a.size
        cst = _gd_constants(a)
        ebar = _expected_length(a)
        guess = pos + np.concatenate(([0.0], np.cumsum(mc * ebar)[:-1]))
        k = np.arange(K)
        margin = int(5.0 * 0.56 * np.sqrt(mc * K) + 0.0005 * mc * K) + 64
        width = int(mc * ebar.max()) + 2 * margin + 64
        w0 = np.maximum(np.floor(guess).astype(np.int64) - margin, pos)
        w0[0] = pos
        slack = 4096
        buf, b0 = U.view(pos, int(w0.max()) + width + slack)
        # lengths that hold for every shape of the batch: immediate acceptance (t >= 0) and the
        # squeeze acceptance of the smallest shape, which implies it for all larger ones
        lo, hi = pos - b0, int(w0.max()) + width - b0
        d_max = float(cst[2].max())
        edges = np.linspace(lo, hi, self.n_jobs + 1).astype(np.int64)
        res = self._map(_shared_lengths, [(buf, int(edges[i]), int(edges[i + 1]), d_max)
                                          for i in range(self.n_jobs)])
        L0 = np.concatenate([r[0] for r in res])
        exc = np.concatenate([r[1] + (edges[i] - lo) for i, r in enumerate(res)])
        # per-call tables: the shared lengths, then the exceptions with each call's own shape
        cols = (w0 - b0 - lo)[:, None] + np.arange(width)[None, :]
        L = np.zeros((K, width + self.PAD), dtype=np.int64)       # zero padding absorbs the walk
        L[:, :width] = L0[cols]
        first = np.searchsorted(exc, w0 - b0 - lo)
        last = np.searchsorted(exc, w0 - b0 - lo + width)
        cnt = last - first
        kk = np.repeat(k, cnt)
        xx = exc[np.repeat(first, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))]
        jj = xx - (w0[kk] - b0 - lo)
        used, t8 = self._lengths(buf, lo + xx, [c[kk] for c in cst])
        L[kk, jj] = used
        is8 = ~np.isnan(t8)
        key8 = kk[is8] * width + jj[is8]
        o = np.argsort(key8)
        key8, val8 = key8[o], t8[is8][o]
        n_steps = mc + int(2 * margin / 2.0) + 8
        POS = self._walk(L, n_steps)
        TOK = np.zeros((K, mc), dtype=np.int64)
        exact = []
        for kc in range(K):
            base = int(w0[kc])
            rel = pos - base
            Lk, Pk = L[kc], POS[kc]
            nd = 0
            ok = 0 <= rel < width
            if ok:
                m = int(np.searchsorted(Pk, rel))
                while Pk[m] != rel and nd < mc:
                    TOK[kc, nd] = rel
                    nd += 1
                    rel += int(Lk[rel])
                    if rel >= width:
                        ok = False
                        break
                    m = int(np.searchsorted(Pk, rel))
            if ok and nd < mc:
                need = mc - nd
                if m + need > n_steps or Pk[m + need] >= width:
                    ok = False
                else:
                    TOK[kc, nd:] = Pk[m:m + need]
                    rel = int(Pk[m + need])
            if not ok:
                vals_k, pos = self._exact_call(U, pos, a[kc], scale[kc], mc, cst, kc)
                exact.append((kc, vals_k))
                buf, b0 = U.view(int(w0.min()), int(w0.max()) + width + slack)
                continue
            pos = base + rel
        upos = (w0 - b0)[:, None] + TOK
        keys = TOK + (k * width)[:, None]
        parts = np.array_split(k, min(self.n_jobs, K))
        vals = np.concatenate(self._map(
            lambda ix: _token_values(buf, upos[ix], keys[ix], key8, val8, cst[0][ix], scale[ix]),
            [(ix,) for ix in parts]))
        for kc, v in exact:
            vals[kc] = v
        return vals, pos

    def _batch_region(self, U, pos, a, scale, mc, max_pass: int = 8):
        """All calls of the batch decoded as one chain over a region of the stream.

        Draw lengths that hold for every shape of the batch are evaluated once per position.
        The others (exceptions) depend on the call a draw belongs to, i.e. on its index in the
        chain: they are evaluated for a provisional call, the chain is decoded, and the
        exceptions on the chain whose call differs are evaluated again, until the chain no
        longer changes. Returns ``None`` if that does not settle (the caller then uses the
        per-call decoding)."""
        K = a.size
        N = K * mc
        cst = _gd_constants(a)
        ebar = _expected_length(a)
        W = int(np.sum(mc * ebar) + 12.0 * 0.56 * np.sqrt(N)) + 512
        buf, b0 = U.view(pos, pos + W + 4096)
        lo = pos - b0
        d_max = float(cst[2].max())
        edges = np.linspace(lo, lo + W, self.n_jobs + 1).astype(np.int64)
        res = self._map(_shared_lengths, [(buf, int(edges[i]), int(edges[i + 1]), d_max)
                                          for i in range(self.n_jobs)])
        L = np.zeros(W + self.PAD, dtype=np.int64)
        L[:W] = np.concatenate([r[0] for r in res])
        exc = np.concatenate([r[1] + (edges[i] - lo) for i, r in enumerate(res)])
        # provisional call of each exception from the expected positions of the calls
        bounds = np.cumsum(mc * ebar)
        k_used = np.minimum(np.searchsorted(bounds, exc, side="right"), K - 1)
        t8 = np.full(exc.size, np.nan)

        def evaluate(ix):
            if ix.size:
                kk = k_used[ix]
                used, tt = self._lengths(buf, lo + exc[ix], [c[kk] for c in cst])
                L[exc[ix]] = used
                t8[ix] = tt

        evaluate(np.arange(exc.size))
        chain = _decode_region(L, W)
        onmask = np.zeros(W + self.PAD, dtype=bool)
        onmask[chain] = True
        for _ in range(max_pass):
            if chain.size <= N:
                return None
            # the call of each exception: from the index of the first chain draw at or after it
            idx = np.searchsorted(chain, exc)
            k_new = np.minimum(idx // mc, K - 1)
            changed = np.flatnonzero(k_new != k_used)
            if changed.size == 0:
                break
            old_L = L[exc[changed]].copy()
            k_used[changed] = k_new[changed]
            evaluate(changed)
            on = onmask[exc[changed]] & (idx[changed] <= N)
            moved = exc[changed[(L[exc[changed]] != old_L) & on]]
            if moved.size == 0:
                break
            # local repair: from a draw whose length changed, follow the chain until it lands on
            # the old chain again, and swap the stretch in between
            for x in np.sort(moved).tolist():
                if not onmask[x]:
                    continue
                p = x + int(L[x])
                path = []
                while p < W and not onmask[p]:
                    path.append(p)
                    p += int(L[p])
                if p >= W:
                    return None
                if p > x + 1:
                    onmask[x + 1:p] = False
                if path:
                    onmask[path] = True
            chain = np.flatnonzero(onmask[:W])
        else:
            return None
        tok = chain[:N]
        is8 = ~np.isnan(t8)
        key8, val8 = exc[is8], t8[is8]
        kidx = np.arange(K)
        upos = lo + tok.reshape(K, mc)
        keys = tok.reshape(K, mc)
        parts = np.array_split(kidx, min(self.n_jobs, K))
        vals = np.concatenate(self._map(
            lambda ix: _token_values(buf, upos[ix], keys[ix], key8, val8, cst[0][ix], scale[ix]),
            [(ix,) for ix in parts]))
        return vals, pos + int(chain[N])

    def _exact_call(self, U, pos, a, scale, mc, cst, k):
        """Decode one call from its known start."""
        ck = [np.atleast_1d(c[k]) for c in cst]
        width = int(mc * 3.2) + 256
        while True:
            buf, b0 = U.view(pos, pos + width + 4096)
            p = pos - b0 + np.arange(width)
            used, t8 = _draw_lengths(buf, p, *[np.repeat(c, width) for c in ck])
            L = np.zeros((1, width + self.PAD), dtype=np.int64)
            L[0, :width] = used
            POS = self._walk(L, mc)
            if POS[0, mc] < width:
                break
            width *= 2
        tok = POS[0, :mc]
        is8 = ~np.isnan(t8)
        vals = _token_values(buf, (pos - b0 + tok)[None, :], tok[None, :], np.flatnonzero(is8),
                             t8[is8], ck[0], np.atleast_1d(scale))[0]
        return vals, pos + int(POS[0, mc])

    @staticmethod
    def _walk(L, n_steps):
        """Positions of the first ``n_steps + 1`` draws of every row's chain, from offset 0. The
        zero padding after the window stops a chain that leaves it."""
        K, w1 = L.shape
        cur = np.zeros(K, dtype=np.int64)
        POS = np.empty((n_steps + 1, K), dtype=np.int64)
        Lf = L.ravel()
        off = np.arange(K, dtype=np.int64) * w1
        idx = off.copy()
        for i in range(n_steps + 1):
            POS[i] = cur
            step = Lf[idx]
            cur += step
            idx += step
        return np.ascontiguousarray(POS.T)


def _decode_region(L: np.ndarray, W: int, seg: int = 1024) -> np.ndarray:
    """Offsets of the chain of draws from offset 0 over ``[0, W)``, given the length ``L`` of a
    draw at every offset (``L`` padded with zeros past ``W``).

    Every segment of ``seg`` offsets is decoded from its own first offset, all segments at once;
    then the segments are joined in order: the chain entering a segment follows its own
    lengths until it lands on that segment's decoding, from where the two coincide."""
    nseg = -(-W // seg)
    starts = np.arange(nseg, dtype=np.int64) * seg
    ends = np.minimum(starts + seg, W)
    cur = starts.copy()
    spec = np.zeros(W + 1, dtype=bool)
    act = np.arange(nseg)
    while act.size:
        c = cur[act]
        spec[c] = True
        cur[act] = c + L[c]
        act = act[cur[act] < ends[act]]
    exit_ = cur
    ystart = ends.copy()
    detour = []
    e = 0
    Ll = L
    for j in range(nseg):
        b = int(ends[j])
        if e >= b:
            continue
        while e < b and not spec[e]:
            detour.append(e)
            e += int(Ll[e])
        if e < b:
            ystart[j] = e
            e = int(exit_[j])
        if e >= W:
            break
    on = spec[:W]
    for j in np.flatnonzero(ystart > starts):
        on[starts[j]:ystart[j]] = False
    if detour:
        on[np.asarray(detour, dtype=np.int64)] = True
    return np.flatnonzero(on)


def _shared_lengths(U, lo, hi, d_max):
    """Lengths of draws starting at offsets ``lo..hi-1`` of ``U`` that hold for every shape whose
    ``d`` is at most ``d_max``: 2 (t >= 0) or 3 (squeeze acceptance); the other offsets are
    returned as exceptions (their length is set to 0)."""
    u1 = U[lo:hi]
    L0 = np.full(hi - lo, 2, dtype=np.int64)
    neg = np.flatnonzero(u1 < 0.5)                        # t < 0 exactly when u1 < 1/2
    pp = (np.floor(BIG * u1[neg]) + U[lo + 1 + neg]) / BIG
    du = d_max * U[lo + 2 + neg]
    # t > -1 when pp > Phi(-1) (with room), so t^3 > -1 >= d * u accepts without computing t
    easy = (pp > 0.1586553) & (du <= -1.0)
    hard = np.flatnonzero(~easy)
    t = qnorm(pp[hard])
    rej = hard[du[hard] > t * t * t]
    L0[neg] = 3
    exc = neg[rej]
    L0[exc] = 0
    return L0, exc


def _token_values(U, upos, keys, key8, val8, s, scale):
    """Draw values of the tokens at offsets ``upos`` (K x mc) of ``U``; draws accepted in steps
    8-11 (``keys`` found in the sorted ``key8``) take their ``t`` from ``val8``."""
    s = np.asarray(s, dtype=np.float64)[:, None]
    scale = np.asarray(scale, dtype=np.float64)[:, None]
    pp = (np.floor(BIG * U[upos]) + U[upos + 1]) / BIG
    t = qnorm(pp.ravel()).reshape(pp.shape)
    x = s + 0.5 * t
    vals = scale * (x * x)
    if key8.size:
        lo, hi = keys[:, 0], keys[:, -1]
        cand = np.flatnonzero((np.searchsorted(key8, hi, side="right") - np.searchsorted(key8, lo)) > 0)
        if cand.size:
            kc = keys[cand]
            j = np.minimum(np.searchsorted(key8, kc), key8.size - 1)
            hit = key8[j] == kc
            r, cc = np.nonzero(hit)
            x8 = s[cand[r], 0] + 0.5 * val8[j[hit]]
            vals[cand[r], cc] = scale[cand[r], 0] * x8 * x8
    return vals


def _r_row_means(v: np.ndarray) -> np.ndarray:
    """R ``mean`` of each row (long double, in order, with the correction pass)."""
    n = v.shape[1]
    s = v.sum(axis=1, dtype=np.longdouble) / n
    t = (v.astype(np.longdouble) - s[:, None]).sum(axis=1)
    return (s + t / n).astype(np.float64)
