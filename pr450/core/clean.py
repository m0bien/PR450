"""
CLEAN: iterative cancellation of strong targets in the time domain.

A strong target drags every waveform artefact along with it: range and
Doppler sidelobes, the LTE CRS-comb ghosts (+-6.6 km), the NR PBCH-DMRS
comb ghosts (+-5 km) and the SSB 50 Hz Doppler comb. Windows only shape
the generic sidelobes; removing the target itself removes all of them.

Each iteration:
  1. range-Doppler map of the current residual, CFAR detection
  2. take the strongest detection (always a real target: no sidelobe
     exceeds its parent); stop if it is below `threshold_db`
  3. refine its delay / Doppler by parabolic interpolation
  4. least-squares fit and subtract its echo from the surveillance signal
     with the dictionary

        ref[t - (d0 + i)] * exp(j 2 pi f0 t) * tau^p,
        i = -half_taps .. half_taps,  p = 0, 1, 2,  tau = (t - T/2) / T

     The integer taps synthesise any fractional delay (the signal is
     oversampled: 4.5 MHz occupied of 7.68 MHz), and the quadratic phase
     polynomial absorbs the residual Doppler error of the interpolation.

Interaction with ECA: ECA fitted with targets present also removes the
zero-Doppler projection of every echo, leaving a negative copy of the
target at 0 Hz that a Doppler-shifted model cannot represent (and that
carries the NR 50 Hz SSB comb with it). Pass `canceller` (e.g. a closure
around eca) together with the *raw* surveillance signal: echoes are then
estimated on the cancelled residual but subtracted from the raw signal,
and the canceller is re-run on the target-free surveillance each iteration.

Subtracted components are returned as detections, so a target removed by
CLEAN is still reported; it just no longer produces ghosts or masks
weaker neighbours.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import solve

from .caf import RDMap, caf_batched
from .cfar import Detection, detect


@dataclass
class CleanComponent:
    delay_samples: float
    doppler_hz: float
    snr_db: float
    residual_drop_db: float


@dataclass
class CleanResult:
    residual: np.ndarray
    components: list[CleanComponent]
    rd: RDMap
    detections: list[Detection]


def _parabolic_offset(p_m1: float, p0: float, p_p1: float) -> float:
    den = p_m1 - 2 * p0 + p_p1
    return 0.0 if den >= 0 else float(np.clip(0.5 * (p_m1 - p_p1) / den, -0.5, 0.5))


def refine_peak(rd: RDMap, det: Detection) -> tuple[float, float]:
    """Sub-bin delay [samples] and Doppler [Hz] of a detection via parabolas on log power."""
    p = 10 * np.log10(rd.power() + 1e-30)
    i, j = det.delay_idx, det.doppler_idx
    di = _parabolic_offset(p[i - 1, j], p[i, j], p[i + 1, j]) if 0 < i < p.shape[0] - 1 else 0.0
    dj = _parabolic_offset(p[i, j - 1], p[i, j], p[i, j + 1]) if 0 < j < p.shape[1] - 1 else 0.0
    fs = 1.0 / (rd.delay_s[1] - rd.delay_s[0])
    df = rd.doppler_hz[1] - rd.doppler_hz[0]
    return (rd.delay_s[i] * fs) + di, rd.doppler_hz[j] + dj * df


def fit_echo(
    surv: np.ndarray,
    ref: np.ndarray,
    fs: float,
    delay_samples: float,
    doppler_hz: float,
    half_taps: int = 8,
    poly_order: int = 2,
    chunk: int = 1 << 16,
    diag_load: float = 1e-9,
) -> np.ndarray:
    n = min(len(surv), len(ref))
    base = int(round(delay_samples)) - half_taps
    n_taps = 2 * half_taps + 1
    # padded[j] = ref[j - lead]; row t of the window view holds ref[t - base - n_taps + 1 .. t - base]
    lead = max(0, base + n_taps - 1)
    tail = max(0, -base)
    padded = np.concatenate([np.zeros(lead, dtype=complex), ref[:n], np.zeros(tail, dtype=complex)])
    view = sliding_window_view(padded, n_taps)
    offset = lead - base - n_taps + 1
    t_norm = (np.arange(n) - n / 2) / n

    def dictionary(s: int, e: int) -> np.ndarray:
        d = view[s + offset:e + offset, ::-1]  # column i = ref[t - (base + i)]
        d = d * np.exp(2j * np.pi * doppler_hz * np.arange(s, e) / fs)[:, None]
        return np.concatenate([d * (t_norm[s:e, None] ** p) for p in range(poly_order + 1)], axis=1)

    m = n_taps * (poly_order + 1)
    gram = np.zeros((m, m), dtype=complex)
    cross = np.zeros(m, dtype=complex)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d = dictionary(s, e)
        gram += d.conj().T @ d
        cross += d.conj().T @ surv[s:e]
    gram += diag_load * np.real(np.trace(gram)) / m * np.eye(m)
    w = solve(gram, cross, assume_a="pos")

    echo = np.empty(n, dtype=complex)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        echo[s:e] = dictionary(s, e) @ w
    return echo


def subtract_echo(surv: np.ndarray, ref: np.ndarray, fs: float, delay_samples: float, doppler_hz: float,
                  **kwargs) -> np.ndarray:
    n = min(len(surv), len(ref))
    return surv[:n] - fit_echo(surv, ref, fs, delay_samples, doppler_hz, **kwargs)


def clean(
    surv: np.ndarray,
    ref: np.ndarray,
    fs: float,
    max_delay_samples: int,
    caf_kwargs: dict | None = None,
    detect_kwargs: dict | None = None,
    threshold_db: float = 18.0,
    max_iter: int = 12,
    half_taps: int = 8,
    canceller: Callable[[np.ndarray], np.ndarray] | None = None,
    refits: int = 1,
) -> CleanResult:
    """See the module docstring. `refits` extra fits of the same target are made
    on the re-cancelled residual: fitting on the ECA output with an unprojected
    dictionary under-estimates the echo by its zero-Doppler projection, which
    left ~-37 dB of a 5.7 Hz target after one fit; one refit reaches ~-52 dB.
    Targets within ~1/T of zero Doppler stay partly inside ECA's notch (~-25 dB).
    """
    caf_kwargs = caf_kwargs or {}
    detect_kwargs = detect_kwargs or {}
    raw = surv
    x = canceller(raw) if canceller is not None else surv
    components: list[CleanComponent] = []
    pending = None  # (component, delay_idx, doppler_idx, power before subtraction)
    while True:
        rd = caf_batched(x, ref, fs, max_delay_samples, **caf_kwargs)
        if pending is not None:
            comp, i, j, p_before = pending
            comp.residual_drop_db = float(10 * np.log10(p_before / max(rd.power()[i, j], 1e-30)))
            pending = None
        dets = detect(rd, **detect_kwargs)
        if not dets or dets[0].snr_db < threshold_db or len(components) >= max_iter:
            break
        strongest = dets[0]
        d0, f0 = refine_peak(rd, strongest)
        for _ in range(1 + (refits if canceller is not None else 0)):
            echo = fit_echo(x, ref, fs, d0, f0, half_taps=half_taps)
            if canceller is not None:
                raw = raw[:len(echo)] - echo
                x = canceller(raw)
            else:
                x = x[:len(echo)] - echo
        comp = CleanComponent(d0, f0, strongest.snr_db, float("nan"))
        components.append(comp)
        pending = (comp, strongest.delay_idx, strongest.doppler_idx,
                   rd.power()[strongest.delay_idx, strongest.doppler_idx])

    n_delay, n_doppler = rd.data.shape
    reported = [
        Detection(
            delay_s=c.delay_samples / fs,
            bistatic_range_m=c.delay_samples / fs * 299_792_458.0,
            doppler_hz=c.doppler_hz,
            snr_db=c.snr_db,
            delay_idx=int(np.clip(round(c.delay_samples), 0, n_delay - 1)),
            doppler_idx=int(np.clip(np.argmin(np.abs(rd.doppler_hz - c.doppler_hz)), 0, n_doppler - 1)),
        )
        for c in components
    ]
    return CleanResult(residual=x, components=components, rd=rd, detections=reported + dets)
