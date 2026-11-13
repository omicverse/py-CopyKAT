"""mixtools ``normalmixEM`` as CopyKAT calls it: three components with a common standard deviation
(``arbvar = FALSE``), starting from given weights, means and sigma, run on several vectors at once.

Each EM iteration follows mixtools: ``lambda = colMeans(post)``; each mean is
``sum(post[, j] * x) / (n * lambda[j])``; the common sigma is
``sqrt(sum(post * res) / (n * sum(lambda)))``; the posteriors, squared residuals and
log-likelihood come from mixtools' ``normpost`` C routine (scaled by the component with the
smallest standardised residual, the log-likelihood added in order). The loop stops when the
log-likelihood gain is at most ``epsilon`` or after ``maxit`` iterations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LD = np.longdouble
_HALF_LOG_2PI = 0.91893853320467274178


@dataclass
class MixFit:
    lam: np.ndarray       # (B, 3)
    mu: np.ndarray        # (B, 3)
    sigma: np.ndarray     # (B,)
    loglik: np.ndarray    # (B,)
    iterations: np.ndarray


def _normpost(x, mu, sigma, lam):
    """mixtools ``normpost`` for B vectors at once: posteriors and squared residuals (B, 3, n) and
    the log-likelihood (B,)."""
    B, n = x.shape
    lsr = lam / sigma[:, None]                                   # LamSigRatio (B, 3)
    llsr = np.log(lsr)
    res2 = x[:, None, :] - mu[:, :, None]
    res2 *= res2
    den = (2.0 * sigma) * sigma
    work = res2 / den[:, None, None]
    w0, w1, w2 = work[:, 0], work[:, 1], work[:, 2]
    # first component with the smallest standardised residual (j == 0 || r < min)
    lt1 = w1 < w0
    mn = np.where(lt1, w1, w0)
    lt2 = w2 < mn
    mn = np.where(lt2, w2, mn)
    minj = np.where(lt2, 2, np.where(lt1, 1, 0))
    rb = np.arange(B)[:, None]
    lsr_min = lsr[rb, minj]
    post = np.empty_like(work)
    rowsum = np.ones((B, n))
    for j in range(3):
        wj = (lsr[:, j][:, None] / lsr_min) * np.exp(mn - work[:, j])
        is_min = minj == j
        wj[is_min] = 1.0
        post[:, j] = wj
        rowsum += np.where(is_min, 0.0, wj)
    post /= rowsum[:, None, :]
    terms = (np.log(rowsum) - mn) + llsr[rb, minj]
    terms[:, 0] += -float(n) * _HALF_LOG_2PI
    ll = np.cumsum(terms, axis=1)[:, -1]
    return post, res2, ll


def normalmix3(x: np.ndarray, sigma0: np.ndarray, mu0=(-0.2, 0.0, 0.2), lam0=None,
               epsilon: float = 1e-8, maxit: int = 1000) -> MixFit:
    """``normalmixEM(x[b], lambda = rep(1, 3)/3, mu = mu0, sigma = sigma0[b], arbvar = FALSE,
    ECM = FALSE, maxit = maxit)`` for each row ``x[b]``."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    B, n = x.shape
    lam_init = np.full(3, 1.0) / 3.0 if lam0 is None else np.asarray(lam0, dtype=np.float64)
    lam_init = lam_init / lam_init.sum()
    lam = np.tile(lam_init, (B, 1))
    mu = np.tile(np.asarray(mu0, dtype=np.float64), (B, 1))
    sigma = np.asarray(sigma0, dtype=np.float64).reshape(B).copy()
    post, res, ll = _normpost(x, mu, sigma, lam)
    iters = np.zeros(B, dtype=np.int64)
    act = np.ones(B, dtype=bool)
    while act.any():
        a = np.flatnonzero(act)
        xa, pa, ra = x[a], post[a], res[a]
        lam_a = (pa.sum(axis=2, dtype=_LD) / n).astype(np.float64)
        mu_a = (pa * xa[:, None, :]).sum(axis=2, dtype=_LD).astype(np.float64) / (float(n) * lam_a)
        num = (pa * ra).reshape(a.size, 3 * n).sum(axis=1, dtype=_LD)
        sl = lam_a.astype(_LD)
        sl = (sl[:, 0] + sl[:, 1]) + sl[:, 2]
        sig_a = np.sqrt(num.astype(np.float64) / (float(n) * sl.astype(np.float64)))
        if np.any(sig_a < 1e-8):
            raise FloatingPointError("normalmixEM: a variance went to zero (R restarts at random)")
        p2, r2, ll2 = _normpost(xa, mu_a, sig_a, lam_a)
        diff = ll2 - ll[a]
        lam[a], mu[a], sigma[a] = lam_a, mu_a, sig_a
        post[a], res[a], ll[a] = p2, r2, ll2
        iters[a] += 1
        act[a] = (diff > epsilon) & (iters[a] < maxit)
    return MixFit(lam=lam, mu=mu, sigma=sigma, loglik=ll, iterations=iters)
