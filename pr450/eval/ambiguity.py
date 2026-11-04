"""
Self-ambiguity analysis and waveform statistics (plan Phase 2b).

|chi(tau, f)| is symmetric under (tau, f) -> (-tau, -f), so positive delays
with both Doppler signs cover the whole function.

Metrics, all relative to the zero-delay / zero-Doppler peak:
  PSLR   -- highest sidelobe outside the mainlobe box
  ISLR   -- integrated sidelobe energy over mainlobe energy
  floor  -- median sidelobe level (the "pedestal" a weak target must beat)
  peaks  -- strongest local sidelobe maxima, i.e. the deterministic artefacts
            (CP replica, CRS comb, SSB/PSS periodicity ...)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..core.caf import RDMap, caf_batched
from ..waveform.base import RE_EMPTY, Waveform


def self_ambiguity(
    wf: Waveform,
    max_delay_s: float = 100e-6,
    doppler_max_hz: float = 600.0,
    batch_len: int = 512,
    window: str | None = "hann",
) -> RDMap:
    max_delay = int(np.ceil(max_delay_s * wf.fs))
    return caf_batched(wf.iq, wf.iq, wf.fs, max_delay, batch_len, doppler_max_hz, window)


def af_metrics(rd: RDMap, mainlobe: tuple[int, int] = (2, 3), n_peaks: int = 8, peak_size: int = 5) -> dict:
    power = rd.power()
    i0, j0 = rd.cell(0.0, 0.0)
    peak = power[i0, j0]
    p = power / peak

    main = np.zeros(p.shape, dtype=bool)
    main[max(0, i0 - mainlobe[0]):i0 + mainlobe[0] + 1, max(0, j0 - mainlobe[1]):j0 + mainlobe[1] + 1] = True
    side = p[~main]

    local_max = (p == ndimage.maximum_filter(p, size=peak_size)) & ~main
    idx = np.argwhere(local_max)
    order = np.argsort(p[local_max])[::-1][:n_peaks]
    peaks = [
        {
            "delay_us": float(rd.delay_s[i] * 1e6),
            "bistatic_km": float(rd.bistatic_range_m[i] / 1e3),
            "doppler_hz": float(rd.doppler_hz[j]),
            "level_db": float(10 * np.log10(p[i, j])),
        }
        for i, j in idx[order]
    ]
    return {
        "pslr_db": float(10 * np.log10(side.max())),
        "islr_db": float(10 * np.log10(side.sum() / p[main].sum())),
        "floor_db": float(10 * np.log10(np.median(side))),
        "peaks": peaks,
    }


def waveform_stats(wf: Waveform) -> dict:
    """Load-related properties that drive passive radar performance."""
    occupied = wf.re_type != RE_EMPTY
    known = wf.known_mask()
    energy = np.abs(wf.grid) ** 2
    sym_power = energy.sum(axis=1)
    iq_power = np.abs(wf.iq) ** 2
    return {
        "occupied_fraction": float(occupied.mean()),
        "known_energy_fraction": float(energy[known].sum() / max(energy.sum(), 1e-30)),
        "mean_power_db": float(10 * np.log10(iq_power.mean())),
        "symbol_power_cv": float(sym_power.std() / max(sym_power.mean(), 1e-30)),
        "papr_db": float(10 * np.log10(iq_power.max() / iq_power.mean())),
    }
