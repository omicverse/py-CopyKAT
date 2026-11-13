"""Pieces of base R with R's arithmetic.

``mean`` adds in long double and corrects with a second pass (summary.c); ``var``/``sd`` use that
mean and a long-double sum of squares (cov.c); ``median`` averages the two middle values for even
lengths; ``qnorm`` is R's ``qnorm5`` (Wichura's AS 241).
"""
from __future__ import annotations

import numpy as np

_LD = np.longdouble


def _seq_sum(m: np.ndarray, axis: int) -> np.ndarray:
    """Long-double sum along ``axis``, adding the slices in order (R's LDOUBLE loops)."""
    m = np.moveaxis(np.asarray(m), axis, 0)
    s = np.zeros(m.shape[1:], dtype=_LD)
    for i in range(m.shape[0]):
        s += m[i]
    return s


def r_means(m: np.ndarray, axis: int = 0) -> np.ndarray:
    """``apply(m, axis + 1 == 2 ? 2 : 1, mean)``: R ``mean`` of each slice along ``axis``."""
    m = np.asarray(m, dtype=np.float64)
    n = m.shape[axis]
    s = _seq_sum(m, axis) / n
    ok = np.isfinite(s.astype(np.float64))
    if ok.any():
        d = np.moveaxis(m, axis, 0).astype(_LD) - s
        t = np.zeros_like(s)
        for i in range(d.shape[0]):
            t += d[i]
        s = np.where(ok, s + t / n, s)
    return s.astype(np.float64)


def r_mean(x) -> float:
    """``mean(x)`` for a double vector."""
    return float(r_means(np.asarray(x, dtype=np.float64).reshape(-1, 1), 0)[0])


def r_sds(m: np.ndarray, axis: int = 0) -> np.ndarray:
    """``sd`` of each slice along ``axis`` (cov.c: long-double mean, then the squares added in
    long double and divided by n - 1)."""
    m = np.asarray(m, dtype=np.float64)
    n = m.shape[axis]
    mu = r_means(m, axis)
    d = np.moveaxis(m, axis, 0) - mu
    ss = _seq_sum(d * d, 0)
    return np.sqrt((ss / (n - 1)).astype(np.float64))


def r_sd(x) -> float:
    """``sd(x)``."""
    return float(r_sds(np.asarray(x, dtype=np.float64).reshape(-1, 1), 0)[0])


def r_medians(m: np.ndarray, axis: int = 0) -> np.ndarray:
    """``median`` of each slice along ``axis`` (the mean of the two middle values for even n)."""
    return np.median(np.asarray(m, dtype=np.float64), axis=axis)


def qnorm(p, log=np.log) -> np.ndarray:
    """``qnorm(p)`` (lower tail): R's ``qnorm5``. ``p`` must lie in (0, 1).

    ``log`` evaluates the tails (|p - 0.5| > 0.425); NumPy's ``log`` can differ from the C library's
    in the last bit."""
    p = np.asarray(p, dtype=np.float64)
    q = p - 0.5
    val = np.empty_like(p)
    c = np.abs(q) <= 0.425
    if c.any():
        qq = q[c]
        r = 0.180625 - qq * qq
        val[c] = (qq * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r
                              + 67265.770927008700853) * r + 45921.953931549871457) * r
                            + 13731.693765509461125) * r + 1971.5909503065514427) * r
                          + 133.14166789178437745) * r + 3.387132872796366608)
                  / (((((((r * 5226.495278852854561 + 28729.085735721942674) * r
                          + 39307.89580009271061) * r + 21213.794301586595867) * r
                        + 5394.1960214247511077) * r + 687.1870074920579083) * r
                      + 42.313330701600911252) * r + 1.0))
    t = ~c
    if t.any():
        qt, pt = q[t], p[t]
        r = np.sqrt(-log(np.where(qt > 0, 1.0 - pt, pt)))
        out = np.empty_like(r)
        a = r <= 5.0
        if a.any():
            ra = r[a] - 1.6
            out[a] = ((((((((ra * 7.7454501427834140764e-4 + 0.0227238449892691845833) * ra
                            + 0.24178072517745061177) * ra + 1.27045825245236838258) * ra
                          + 3.64784832476320460504) * ra + 5.7694972214606914055) * ra
                        + 4.6303378461565452959) * ra + 1.42343711074968357734)
                      / (((((((ra * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * ra
                              + 0.0151986665636164571966) * ra + 0.14810397642748007459) * ra
                            + 0.68976733498510000455) * ra + 1.6763848301838038494) * ra
                          + 2.05319162663775882187) * ra + 1.0))
        b = ~a
        if b.any():
            rb = r[b] - 5.0
            out[b] = ((((((((rb * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * rb
                            + 0.0012426609473880784386) * rb + 0.026532189526576123093) * rb
                          + 0.29656057182850489123) * rb + 1.7848265399172913358) * rb
                        + 5.4637849111641143699) * rb + 6.6579046435011037772)
                      / (((((((rb * 2.04426310338993978564e-15 + 1.4215117583164458887e-7) * rb
                              + 1.8463183175100546818e-5) * rb + 7.868691311456132591e-4) * rb
                            + 0.0148753612908506148525) * rb + 0.13692988092273580531) * rb
                          + 0.59983220655588793769) * rb + 1.0))
        val[t] = np.where(qt < 0.0, -out, out)
    return val
