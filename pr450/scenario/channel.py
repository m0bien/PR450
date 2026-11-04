"""
Two-channel (reference + surveillance) receiver simulation.

Amplitudes are in units of the receiver noise: complex noise has unit
variance per sample over the full sample rate, and a component received at
power P has amplitude sqrt(P / N) for a *fully loaded* carrier. The
waveform's own power already drops at lower load, so the actual SNR follows
the load automatically.

Delays are applied with a windowed-sinc fractional-delay FIR. All
zero-Doppler paths (direct path + clutter) share one FIR so that a dense
clutter field costs a single convolution. Each moving target gets its own
FIR followed by a Doppler phase ramp. The delay is held constant over the
record (no range migration), which holds for v * T_cpi << c / B (~67 m).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import oaconvolve

from ..waveform.base import Waveform
from .geometry import (
    C,
    Scenario,
    bistatic_echo_dbm,
    bistatic_range_doppler,
    friis_dbm,
    noise_power_dbm,
    random_clutter,
)

FIR_HALF = 16


@dataclass
class TargetTruth:
    name: str
    bistatic_range_m: float
    delay_s: float
    delay_samples: float
    doppler_hz: float
    snr_db: float  # per sample, relative to noise, full-load equivalent


@dataclass
class SimResult:
    ref: np.ndarray
    surv: np.ndarray
    fs: float
    truth: list[TargetTruth]
    levels_db: dict = field(default_factory=dict)


def fractional_delay_fir(delays_samples, gains) -> np.ndarray:
    """Windowed-sinc FIR for a sum of delayed paths; output is delayed by FIR_HALF samples (apply_fir removes it).
    Delays must be >= 0 (relative to the direct path); an empty path list gives an all-zero FIR."""
    delays_samples = np.atleast_1d(np.asarray(delays_samples, dtype=float))
    gains = np.atleast_1d(np.asarray(gains, dtype=complex))
    if delays_samples.size == 0:
        return np.zeros(2 * FIR_HALF + 1, dtype=complex)
    if delays_samples.min() < 0:
        raise ValueError("fractional_delay_fir: delays must be >= 0 samples")
    n = np.arange(int(np.ceil(delays_samples.max())) + 2 * FIR_HALF + 1)
    h = np.zeros(len(n), dtype=complex)
    for d, g in zip(delays_samples, gains):
        x = n - FIR_HALF - d
        win = np.where(np.abs(x) <= FIR_HALF, 0.5 * (1 + np.cos(np.pi * x / FIR_HALF)), 0.0)
        h += g * np.sinc(x) * win
    return h


def apply_fir(s: np.ndarray, h: np.ndarray) -> np.ndarray:
    return oaconvolve(s, h)[FIR_HALF:FIR_HALF + len(s)]


def quantize(x: np.ndarray, bits: int, backoff_db: float, rms: float | None = None) -> tuple[np.ndarray, float]:
    """Mid-rise uniform quantiser, full scale at backoff_db above `rms` (default: the record's own RMS).

    Passing the full-load-equivalent RMS models a receiver whose gain is set
    for busy-hour traffic: a lightly loaded or bursty carrier then uses less
    of the ADC range instead of being clipped by an over-eager AGC.
    """
    if rms is None:
        rms = np.sqrt(np.mean(np.abs(x) ** 2))
    full_scale = rms * 10 ** (backoff_db / 20)
    step = 2 * full_scale / 2**bits
    top = 2 ** (bits - 1) - 0.5

    def q(v):
        return np.clip(np.floor(v / step) + 0.5, -top, top) * step

    clipped = np.mean((np.abs(x.real) > full_scale) | (np.abs(x.imag) > full_scale))
    return q(x.real) + 1j * q(x.imag), float(clipped)


def simulate(wf: Waveform, sc: Scenario, seed: int = 0) -> SimResult:
    rng = np.random.default_rng(seed)
    fs = wf.fs
    lam = C / wf.fc_hz
    n0_dbm = noise_power_dbm(fs, sc.noise_figure_db, sc.temperature_k)

    def amp(p_dbm: float) -> float:
        return 10 ** ((p_dbm - n0_dbm) / 20)

    s = wf.iq
    n = len(s)
    baseline = sc.baseline_m

    # Reference channel: direct path + weak multipath + noise
    p_ref = friis_dbm(sc.eirp_dbm, sc.ref_gain_dbi, baseline, lam)
    ref_delays = [0.0] + [rb / C * fs for rb, _ in sc.ref_multipath]
    ref_gains = [amp(p_ref)] + [
        amp(p_ref + rel_db) * np.exp(2j * np.pi * rng.random()) for _, rel_db in sc.ref_multipath
    ]
    ref = apply_fir(s, fractional_delay_fir(ref_delays, ref_gains))

    # Surveillance channel: attenuated direct path + clutter (zero Doppler)
    p_dpi = friis_dbm(sc.eirp_dbm, sc.surv_gain_dbi - sc.dpi_suppression_db, baseline, lam)
    delays = [0.0]
    gains = [amp(p_dpi) * np.exp(2j * np.pi * rng.random())]
    clutter_cnr = []
    for pos, rcs in random_clutter(sc.clutter, sc.rx_position_m):
        rb, _, r_tx, r_rx = bistatic_range_doppler(sc.tx_position_m, sc.rx_position_m, pos, (0, 0, 0), lam)
        p = bistatic_echo_dbm(sc.eirp_dbm, sc.surv_gain_dbi, rcs, r_tx, r_rx, lam)
        delays.append(rb / C * fs)
        gains.append(amp(p) * np.exp(2j * np.pi * rng.random()))
        clutter_cnr.append(p - n0_dbm)
    surv = apply_fir(s, fractional_delay_fir(delays, gains))

    # Moving targets
    t = np.arange(n) / fs
    truth = []
    for tgt in sc.targets:
        rb, fd, r_tx, r_rx = bistatic_range_doppler(
            sc.tx_position_m, sc.rx_position_m, tgt.position_m, tgt.velocity_mps, lam
        )
        p = bistatic_echo_dbm(sc.eirp_dbm, sc.surv_gain_dbi, tgt.rcs_dbsm, r_tx, r_rx, lam)
        d = rb / C * fs
        echo = apply_fir(s, fractional_delay_fir([d], [amp(p) * np.exp(2j * np.pi * rng.random())]))
        surv += echo * np.exp(2j * np.pi * fd * t)
        truth.append(TargetTruth(tgt.name, rb, rb / C, d, fd, p - n0_dbm))
    for inj in sc.injected:
        d = inj.bistatic_range_m / C * fs
        echo = apply_fir(s, fractional_delay_fir([d], [10 ** (inj.snr_db / 20) * np.exp(2j * np.pi * rng.random())]))
        surv += echo * np.exp(2j * np.pi * inj.doppler_hz * t)
        truth.append(TargetTruth(inj.name, inj.bistatic_range_m, inj.bistatic_range_m / C, d, inj.doppler_hz, inj.snr_db))

    def noise():
        return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)

    ref = ref + noise()
    surv = surv + noise()

    levels = {
        "noise_dbm": n0_dbm,
        "ref_dnr_db": p_ref - n0_dbm,
        "surv_dnr_db": p_dpi - n0_dbm,
        "clutter_cnr_max_db": max(clutter_cnr) if clutter_cnr else float("-inf"),
        "clutter_max_delay_samples": max(delays),
    }
    if sc.adc_bits:
        # Gain set for a fully loaded carrier: sum of component powers + noise
        ref_rms = np.sqrt(np.sum(np.abs(ref_gains) ** 2) + 1)
        surv_rms = np.sqrt(np.sum(np.abs(gains) ** 2) + sum(10 ** (t.snr_db / 10) for t in truth) + 1)
        ref, levels["ref_clip_fraction"] = quantize(ref, sc.adc_bits, sc.adc_backoff_db, ref_rms)
        surv, levels["surv_clip_fraction"] = quantize(surv, sc.adc_bits, sc.adc_backoff_db, surv_rms)
    return SimResult(ref=ref, surv=surv, fs=fs, truth=truth, levels_db=levels)
