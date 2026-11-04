"""
2-D cell-averaging CFAR on a range-Doppler power map, plus detection extraction.

The threshold multiplier assumes exponentially distributed (square-law,
Gaussian-noise) cell powers: alpha = N (Pfa^(-1/N) - 1) for N training cells.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .caf import RDMap


@dataclass
class Detection:
    delay_s: float
    bistatic_range_m: float
    doppler_hz: float
    snr_db: float
    delay_idx: int
    doppler_idx: int


def ca_cfar(
    power: np.ndarray,
    guard: tuple[int, int] = (2, 3),
    train: tuple[int, int] = (4, 8),
    pfa: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (detection mask, local noise estimate)."""
    gd, gf = guard
    td, tf = train
    kernel = np.ones((2 * (gd + td) + 1, 2 * (gf + tf) + 1))
    kernel[td:td + 2 * gd + 1, tf:tf + 2 * gf + 1] = 0
    n_train = kernel.sum()
    noise = ndimage.convolve(power, kernel / n_train, mode="nearest")
    alpha = n_train * (pfa ** (-1 / n_train) - 1)
    return power > alpha * noise, noise


def detect(
    rd: RDMap,
    guard: tuple[int, int] = (2, 3),
    train: tuple[int, int] = (4, 8),
    pfa: float = 1e-6,
    min_delay_samples: int = 0,
    exclude_zero_doppler_hz: float = 0.0,
) -> list[Detection]:
    power = rd.power()
    mask, noise = ca_cfar(power, guard, train, pfa)
    mask[:min_delay_samples] = False
    if exclude_zero_doppler_hz > 0:
        mask[:, np.abs(rd.doppler_hz) <= exclude_zero_doppler_hz] = False

    labels, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        cells = np.argwhere(labels == i)
        best = cells[np.argmax(power[cells[:, 0], cells[:, 1]])]
        d, f = int(best[0]), int(best[1])
        out.append(
            Detection(
                delay_s=float(rd.delay_s[d]),
                bistatic_range_m=float(rd.bistatic_range_m[d]),
                doppler_hz=float(rd.doppler_hz[f]),
                snr_db=float(10 * np.log10(power[d, f] / noise[d, f])),
                delay_idx=d,
                doppler_idx=f,
            )
        )
    return sorted(out, key=lambda x: -x.snr_db)
