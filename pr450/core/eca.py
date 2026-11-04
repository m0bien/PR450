"""
Extensive Cancellation Algorithm: classic ECA, ECA-B and ECA-S in one routine.

The surveillance signal is modelled as a linear combination of delayed (and
optionally Doppler-shifted) copies of the reference,

    x[n] ~= sum_d sum_{l=-K}^{L-1} w[d, l] * ref[n - l] * exp(j 2 pi f_d n / fs)

and the least-squares fit is subtracted. K = n_lead pre-cursor taps are
needed whenever part of the interference arrives "before" the reference,
e.g. sub-sample clutter delays (band-limited tails extend to negative lags)
or a small ref/surv timing offset. The record is processed in batches
of `batch_len` samples. For each batch, the Gram matrix D^H D and the cross
term D^H x are computed, and the weights used for batch b are solved from
the sum over batches b-window .. b+window:

    window = None  -> one weight vector for the whole record (classic ECA)
    window = 0     -> independent weights per batch (ECA-B)
    window = k > 0 -> sliding average over 2k+1 batches (ECA-S)

ECA-B/ECA-S track slowly time-varying clutter and also widen the cancelled
notch around zero Doppler, so slow targets lose energy. Choose the batch
length with this trade-off in mind: notch width ~ fs / batch_len.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import fftconvolve


@dataclass
class EcaResult:
    output: np.ndarray
    cancellation_db: float
    n_batches: int


class EcaCanceller:
    """ECA with the reference-only work done once.

    The Gram matrices D^H D (and their Cholesky factors) depend only on the
    reference, so they are computed at construction; each call then costs
    only the cross terms D^H x and the subtraction. This matters when the
    same reference is cancelled repeatedly, e.g. inside CLEAN.
    """

    def __init__(
        self,
        ref: np.ndarray,
        n_taps: int,
        batch_len: int = 1 << 15,
        window: int | None = None,
        n_lead: int = 0,
        doppler_offsets_hz: tuple[float, ...] = (0.0,),
        fs: float | None = None,
        diag_load: float = 1e-10,
    ):
        if any(f != 0 for f in doppler_offsets_hz) and fs is None:
            raise ValueError("fs is required when doppler_offsets_hz contains non-zero offsets")
        self.n = n = len(ref)
        self.ref = ref
        self.n_taps, self.n_lead = n_taps, n_lead
        self.window = window
        self.doppler_offsets_hz = tuple(doppler_offsets_hz)
        self.fs = fs
        refpad = np.concatenate([np.zeros(n_taps - 1, dtype=complex), ref, np.zeros(n_lead, dtype=complex)])
        self._taps_view = sliding_window_view(refpad, n_taps + n_lead)  # row t = ref[t-L+1 .. t+K]
        starts = list(range(0, n, batch_len))
        if len(starts) > 1 and n - starts[-1] < batch_len // 2:
            starts.pop()                     # merge a short tail into the previous batch (no ill-posed tail fit)
        self.bounds = [(s, e) for s, e in zip(starts, starts[1:] + [n])]

        def gram(s: int, e: int) -> np.ndarray:
            d = self._dictionary(s, e)
            return d.conj().T @ d

        grams = np.array([gram(s, e) for s, e in self.bounds])

        def factor(g: np.ndarray):
            m = g.shape[0]
            return cho_factor(g + diag_load * np.real(np.trace(g)) / m * np.eye(m))

        if window is None:
            self._factors = [factor(grams.sum(axis=0))]
        else:
            nb = len(self.bounds)
            self._factors = [factor(grams[max(0, b - window):min(nb, b + window + 1)].sum(axis=0)) for b in range(nb)]

    def _dictionary(self, s: int, e: int) -> np.ndarray:
        d = np.ascontiguousarray(self._taps_view[s:e, ::-1])  # columns: ref[t + K] ... ref[t - L + 1]
        if len(self.doppler_offsets_hz) == 1 and self.doppler_offsets_hz[0] == 0:
            return d
        t = np.arange(s, e) / self.fs
        return np.concatenate([d * np.exp(2j * np.pi * f * t)[:, None] for f in self.doppler_offsets_hz], axis=1)

    def __call__(self, surv: np.ndarray) -> EcaResult:
        n = min(self.n, len(surv))
        if len(surv) != self.n:
            raise ValueError(f"surveillance has {len(surv)} samples, the canceller was built for {self.n}")
        surv = surv[:n]
        if self.window is None and self.doppler_offsets_hz == (0.0,):
            return self._call_fast(surv)
        dicts = [self._dictionary(s, e) for s, e in self.bounds]
        crosses = np.array([d.conj().T @ surv[s:e] for d, (s, e) in zip(dicts, self.bounds)])

        out = np.empty(n, dtype=complex)
        if self.window is None:
            w = cho_solve(self._factors[0], crosses.sum(axis=0))
        nb = len(self.bounds)
        for b, ((s, e), d) in enumerate(zip(self.bounds, dicts)):
            if self.window is not None:
                lo, hi = max(0, b - self.window), min(nb, b + self.window + 1)
                w = cho_solve(self._factors[b], crosses[lo:hi].sum(axis=0))
            out[s:e] = surv[s:e] - d @ w

        p_in = np.mean(np.abs(surv) ** 2)
        p_out = np.mean(np.abs(out) ** 2)
        return EcaResult(out, float(10 * np.log10(p_in / p_out)), nb)


    def _call_fast(self, surv: np.ndarray) -> EcaResult:
        """Whole-CPI zero-Doppler ECA: D^H x is a cross-correlation and D w a convolution."""
        n, k, l = self.n, self.n_lead, self.n_taps
        corr = fftconvolve(surv, np.conj(self.ref[::-1]))     # corr[n-1+lag] = sum_t x[t] conj(ref[t-lag])
        lags = np.arange(-k, l)                                # column j <-> lag j - K
        w = cho_solve(self._factors[0], corr[n - 1 + lags])
        x_hat = fftconvolve(self.ref, w)[k:k + n]              # sum_lag w(lag) ref[t - lag]
        out = surv - x_hat
        p_in = np.mean(np.abs(surv) ** 2)
        return EcaResult(out, float(10 * np.log10(p_in / np.mean(np.abs(out) ** 2))), len(self.bounds))


def eca(
    surv: np.ndarray,
    ref: np.ndarray,
    n_taps: int,
    batch_len: int = 1 << 15,
    window: int | None = None,
    n_lead: int = 0,
    doppler_offsets_hz: tuple[float, ...] = (0.0,),
    fs: float | None = None,
    diag_load: float = 1e-10,
) -> EcaResult:
    n = min(len(surv), len(ref))
    canceller = EcaCanceller(ref[:n], n_taps, batch_len, window, n_lead, doppler_offsets_hz, fs, diag_load)
    return canceller(surv[:n])


def eca_b(surv, ref, n_taps, batch_len=1 << 13, **kwargs) -> EcaResult:
    return eca(surv, ref, n_taps, batch_len=batch_len, window=0, **kwargs)


def eca_s(surv, ref, n_taps, batch_len=1 << 13, window=2, **kwargs) -> EcaResult:
    return eca(surv, ref, n_taps, batch_len=batch_len, window=window, **kwargs)
