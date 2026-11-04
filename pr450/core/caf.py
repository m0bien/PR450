"""
Cross-ambiguity function (range-Doppler map), two implementations.

caf_batched -- waveform-agnostic "batches" algorithm. The record is cut into
    batches of batch_len samples; each batch is cross-correlated with the
    reference via FFT for delays 0..max_delay-1, then an FFT across batches
    gives Doppler. Unambiguous Doppler is +-fs / (2 batch_len).

caf_ofdm -- OFDM-native processing on the resource grid. Each surveillance
    symbol is FFT'd with the reference symbol timing, divided by (reciprocal
    filter) or multiplied by the conjugate of (matched filter) the reference
    grid, IFFT'd to a delay profile, and an FFT across symbols gives Doppler.
    The reciprocal filter removes the data-dependent range sidelobes and
    ignores empty REs, which matters for a lightly loaded NR cell. It is only
    inter-symbol-interference free for delays up to the CP (36 samples ~
    4.7 us ~ 1.4 km bistatic); longer delays suffer a gradual ISI loss.

Both return an RDMap with delay along axis 0 and Doppler along axis 1.

Sidelobe control (see results of the false-alarm study):
  window          slow-time (Doppler) taper
  range_window    spectral taper across the occupied band. The occupied
                  spectrum is sharp-edged (300 of 512 bins), so without it the
                  range response has sinc-like sidelobes (-13 dB first, slow
                  decay) that strong targets push above the floor for km.
  normalize       exponent alpha of the per-batch / per-symbol energy
                  equalisation (mean/E)^alpha, 0 = off, so that a
                  bursty (load-dependent) illuminator does not amplitude-
                  modulate the echoes, which spreads them in Doppler.
Window names are anything scipy.signal.get_window accepts, e.g. "hann",
"blackmanharris", ("chebwin", 90), ("taylor", 6, 60).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import fft, fftfreq, fftshift, ifft, next_fast_len
from scipy.signal import get_window

from ..scenario.geometry import C
from ..waveform.base import Waveform, ofdm_demodulate


@dataclass
class RDMap:
    data: np.ndarray
    delay_s: np.ndarray
    doppler_hz: np.ndarray

    @property
    def bistatic_range_m(self) -> np.ndarray:
        return self.delay_s * C

    def power(self) -> np.ndarray:
        return np.abs(self.data) ** 2

    def power_db(self, reference: float | None = None) -> np.ndarray:
        p = self.power()
        ref = p.max() if reference is None else reference
        return 10 * np.log10(p / ref + 1e-30)

    def cell(self, delay_s: float, doppler_hz: float) -> tuple[int, int]:
        return int(np.argmin(np.abs(self.delay_s - delay_s))), int(np.argmin(np.abs(self.doppler_hz - doppler_hz)))


def _window(window, n: int) -> np.ndarray:
    if window in (None, "none", "rect", "boxcar"):
        return np.ones(n)
    return get_window(window, n, fftbins=False)


def spectral_taper(window, freqs_hz: np.ndarray, occupied_bw_hz: float) -> np.ndarray:
    """Window evaluated continuously over [-B/2, B/2], zero outside the occupied band."""
    x = (np.asarray(freqs_hz) + occupied_bw_hz / 2) / occupied_bw_hz
    proto = _window(window, 4097)
    w = np.interp(x, np.linspace(0, 1, len(proto)), proto)
    return np.where((x >= 0) & (x <= 1), w, 0.0)


def _energy_weights(energy: np.ndarray, alpha: float, floor: float = 0.1) -> np.ndarray:
    """Per-batch / per-symbol gains (mean / E)^alpha; slots below floor * mean get weight 0.

    alpha = 0 is the plain matched filter and SNR-optimal; alpha = 1 fully
    equalises the echo amplitude. For alpha >= 0.5 a low-energy slot adds as
    much noise as a full one while carrying almost no signal: with the former
    floor of 1e-3 * mean, CPIs with many partially filled batches lost up to
    ~9 dB (Pd/Pfa study). The 0.1 floor prevents that collapse, but alpha = 0
    remains the recommended setting.
    """
    mean = energy.mean()
    ok = energy > floor * mean
    w = np.zeros_like(energy)
    w[ok] = (mean / energy[ok]) ** alpha
    return w


def _doppler_fft(slow_by_delay: np.ndarray, window, dt: float, doppler_max_hz) -> tuple[np.ndarray, np.ndarray]:
    n = slow_by_delay.shape[0]
    w = _window(window, n)
    spec = fftshift(fft(slow_by_delay * w[:, None], axis=0), axes=0) / np.sum(w)
    f = fftshift(fftfreq(n, d=dt))
    if doppler_max_hz is not None:
        sel = np.abs(f) <= doppler_max_hz
        spec, f = spec[sel], f[sel]
    return spec.T, f


def caf_batched(
    surv: np.ndarray,
    ref: np.ndarray,
    fs: float,
    max_delay_samples: int,
    batch_len: int = 512,
    doppler_max_hz: float | None = None,
    window="hann",
    range_window=None,
    occupied_bw_hz: float | None = None,
    normalize: float = 0.0,
    mode: str = "matched",
    reg: float = 0.1,
    chunk: int = 2048,
) -> RDMap:
    """mode="matched": cross-correlation with the reference (SNR-optimal for noise).
    mode="reciprocal": each batch spectrum is divided by the reference batch spectrum,
    H = Y R* / (|R|^2 + reg * mean|R|^2), i.e. a per-batch channel (CSI) estimate.
    It flattens each batch's random-data spectrum (the source of target self-noise),
    but is exact only for the part of an echo that falls inside its own batch.
    """
    if mode not in ("matched", "reciprocal"):
        raise ValueError("mode must be 'matched' or 'reciprocal'")
    n = min(len(surv), len(ref))
    nb = (n - max_delay_samples) // batch_len
    if nb < 2:
        raise ValueError("record too short for this batch_len / max_delay")
    seg = batch_len + max_delay_samples
    nfft = next_fast_len(seg)
    segments = sliding_window_view(surv[:n], seg)[::batch_len][:nb]

    if range_window is not None:
        if occupied_bw_hz is None:
            raise ValueError("range_window needs occupied_bw_hz")
        taper = spectral_taper(range_window, fftfreq(nfft, d=1 / fs), occupied_bw_hz)
    else:
        taper = None
    freqs = fftfreq(nfft, d=1 / fs)
    band = np.abs(freqs) <= (occupied_bw_hz / 2 if occupied_bw_hz else fs / 2)

    corr = np.empty((nb, max_delay_samples), dtype=np.complex64)
    energy = np.empty(nb)
    for c0 in range(0, nb, chunk):
        c1 = min(nb, c0 + chunk)
        s_f = fft(segments[c0:c1], nfft, axis=1)
        r_batches = ref[c0 * batch_len:c1 * batch_len].reshape(c1 - c0, batch_len)
        energy[c0:c1] = np.sum(np.abs(r_batches) ** 2, axis=1)
        r_f = fft(r_batches, nfft, axis=1)
        if mode == "reciprocal":
            power = np.abs(r_f) ** 2
            level = power[:, band].mean(axis=1, keepdims=True)
            filt = np.where(level > 0, np.conj(r_f) * level / (power + reg * level + 1e-30), 0)
        else:
            filt = np.conj(r_f)
        prod = s_f * filt
        if taper is not None:
            prod *= taper
        corr[c0:c1] = ifft(prod, axis=1)[:, :max_delay_samples] / batch_len
    if normalize:
        corr *= _energy_weights(energy, float(normalize))[:, None].astype(np.float32)

    data, f = _doppler_fft(corr, window, batch_len / fs, doppler_max_hz)
    return RDMap(data, np.arange(max_delay_samples) / fs, f)


def caf_ofdm(
    surv: np.ndarray,
    wf: Waveform,
    ref_grid: np.ndarray,
    max_delay_samples: int,
    mode: str = "reciprocal",
    eps: float = 0.1,
    doppler_max_hz: float | None = None,
    window="hann",
    range_window=None,
    normalize: float = 0.0,
) -> RDMap:
    num = wf.num
    if max_delay_samples > num.nfft:
        raise ValueError("max_delay_samples cannot exceed the FFT size")
    y = ofdm_demodulate(surv, num, wf.sym_starts, wf.cp_lens)
    if mode == "matched":
        h = y * np.conj(ref_grid)
    elif mode == "reciprocal":
        use = np.abs(ref_grid) > eps
        h = np.where(use, y / np.where(use, ref_grid, 1), 0)
    else:
        raise ValueError("mode must be 'matched' or 'reciprocal'")

    if range_window is not None:
        off = num.subcarrier_offsets() * num.scs_hz
        h = h * spectral_taper(range_window, off, (off.max() - off.min()) + num.scs_hz)[None, :]
    if normalize:
        used = np.abs(ref_grid) > eps if mode == "reciprocal" else np.abs(ref_grid) ** 2
        h = h * _energy_weights(used.sum(axis=1).astype(float), float(normalize))[:, None]

    spec = np.zeros((wf.n_symbols, num.nfft), dtype=complex)
    spec[:, num.subcarrier_bins()] = h
    profiles = ifft(spec, axis=1)[:, :max_delay_samples]

    dt = (wf.sym_starts[-1] - wf.sym_starts[0]) / (wf.n_symbols - 1) / wf.fs
    data, f = _doppler_fft(profiles, window, dt, doppler_max_hz)
    return RDMap(data, np.arange(max_delay_samples) / wf.fs, f)
