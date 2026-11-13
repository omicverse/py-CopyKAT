"""R's default random number generator, so that a seed gives the same draws as in R.

R (>= 3.6) defaults: Mersenne-Twister, normals by inversion, ``sample()`` by rejection sampling.
``set_seed`` follows ``RNG_Init`` (RNG.c): the seed is scrambled 50 times with the linear
congruential step ``seed = 69069 * seed + 1`` and the next 625 values fill the generator state
(the first is the position, reset to 624). ``unif_rand`` is MT19937 scaled by 2^-32;
``unif_index`` is ``R_unif_index``. CopyKAT draws through ``rgamma`` (MCMCpack's
``MCpoissongamma``, see :mod:`.rgamma`) and, in cell-line mode, ``rnorm``.
"""
from __future__ import annotations

import math

import numpy as np

from .rbase import qnorm

_clog = np.frompyfunc(math.log, 1, 1)


def clog(x) -> np.ndarray:
    """The C library ``log`` element-wise (R's), as float64."""
    return np.asarray(_clog(np.asarray(x, dtype=np.float64)), dtype=np.float64)

_I2_32M1 = 2.328306437080797e-10      # 1 / (2^32 - 1), RNG.c fixup()


class RRandom:
    """A stream of R's ``runif``/``sample`` draws (Mersenne-Twister, rejection sampling)."""

    def __init__(self, seed: int | None = None):
        self._bits = np.random.MT19937()
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int) -> "RRandom":
        """R ``set.seed(seed)``."""
        s = int(seed) & 0xFFFFFFFF
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        key = np.empty(625, dtype=np.uint32)
        for j in range(625):
            s = (69069 * s + 1) & 0xFFFFFFFF
            key[j] = s
        self._bits.state = {"bit_generator": "MT19937", "state": {"key": key[1:], "pos": 624}}
        return self

    def unif_rand(self, n: int) -> np.ndarray:
        """``n`` draws of R ``unif_rand()``."""
        y = self._bits.random_raw(n).astype(np.float64) * 2.3283064365386963e-10
        y[y <= 0.0] = 0.5 * _I2_32M1
        y[1.0 - y <= 0.0] = 1.0 - 0.5 * _I2_32M1
        return y

    def unif_index(self, dn: int) -> int:
        """R ``R_unif_index(dn)``: an integer in [0, dn) by rejection from ``bits`` random bits."""
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        mask = (1 << bits) - 1
        chunks = bits // 16 + 1
        while True:
            u = np.floor(self.unif_rand(chunks) * 65536).astype(np.int64)
            v = 0
            for c in u:
                v = 65536 * v + int(c)
            v &= mask
            if v < dn:
                return v

    def _sample_replace(self, n: int, size: int) -> np.ndarray:
        """``size`` draws of ``R_unif_index(n)``, vectorised: draw a batch of attempts, keep the
        first ``size`` accepted ones and advance the stream by exactly the uniforms they used."""
        if n <= 1:
            self._bits.random_raw(size)
            return np.zeros(size, dtype=np.int64)
        bits = int(math.ceil(math.log2(n)))
        chunks = bits // 16 + 1
        mask = (1 << bits) - 1
        state = self._bits.state
        attempts = size + size // 2 + 64
        while True:
            u = np.floor(self.unif_rand(attempts * chunks) * 65536).astype(np.int64).reshape(attempts, chunks)
            v = np.zeros(attempts, dtype=np.int64)
            for c in range(chunks):
                v = 65536 * v + u[:, c]
            v &= mask
            ok = np.flatnonzero(v < n)
            if ok.size >= size:
                break
            attempts *= 2
            self._bits.state = state
        self._bits.state = state
        self._bits.random_raw((ok[size - 1] + 1) * chunks)
        return v[ok[:size]]

    def sample(self, n: int, size: int, replace: bool = False) -> np.ndarray:
        """R ``sample.int(n, size, replace)`` as 0-based indices (no hashing, n <= 1e7).

        Without replacement each draw is ``R_unif_index(n - i)`` followed by R's swap; every step
        depends on the previous one, so it runs over plain integers, taking the 32-bit words of
        the stream in batches and returning the unused ones."""
        if replace or size < 2:
            return self._sample_replace(n, size)
        state = self._bits.state
        batch = 2 * size + 1024
        raw = self._bits.random_raw(batch).tolist()
        pos = 0
        pool = list(range(n))
        out = [0] * size
        left = n
        for i in range(size):
            bits = (left - 1).bit_length() if left > 1 else 0
            mask = (1 << bits) - 1
            chunks = bits // 16 + 1
            while True:
                if pos + chunks > len(raw):
                    raw.extend(self._bits.random_raw(batch).tolist())
                v = 0
                for c in range(chunks):
                    v = 65536 * v + (raw[pos + c] >> 16)
                pos += chunks
                v &= mask
                if v < left:
                    break
            out[i] = pool[v]
            left -= 1
            pool[v] = pool[left]
        self._bits.state = state
        self._bits.random_raw(pos)
        return np.asarray(out, dtype=np.int64)

    def norm_rand(self, n: int) -> np.ndarray:
        """``n`` draws of R ``norm_rand()`` by inversion: two uniforms each,
        ``qnorm((floor(2^27 u1) + u2) / 2^27)``."""
        u = self.unif_rand(2 * n).reshape(n, 2)
        big = 134217728.0
        return qnorm((np.floor(big * u[:, 0]) + u[:, 1]) / big, log=clog)

    def rnorm(self, mean, sd) -> np.ndarray:
        """R ``rnorm`` for each (mean, sd) pair, in order. As R, an entry with ``sd == 0``
        returns ``mean`` without drawing."""
        mean = np.asarray(mean, dtype=np.float64)
        sd = np.broadcast_to(np.asarray(sd, dtype=np.float64), mean.shape)
        out = mean.copy()
        draw = sd != 0.0
        out[draw] = mean[draw] + sd[draw] * self.norm_rand(int(draw.sum()))
        return out

    def skip_normals(self, n: int) -> "RRandom":
        """Advance the stream past ``rnorm(n)`` (inversion uses two uniforms per normal)."""
        self._bits.random_raw(2 * n)
        return self
