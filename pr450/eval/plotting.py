"""Matplotlib helpers for range-Doppler maps and ambiguity cuts."""
from __future__ import annotations

import numpy as np

from ..core.caf import RDMap


def plot_rd(
    ax,
    rd: RDMap,
    title: str = "",
    dyn_range_db: float = 60.0,
    reference: float | None = None,
    floor_normalise: bool = False,
    truth=None,
    detections=None,
    cmap: str = "viridis",
):
    """Image of |RD|^2 in dB, bistatic range [km] vs Doppler [Hz].

    floor_normalise=True scales to the median cell (colour = SNR over floor),
    otherwise to `reference` or the map peak.
    """
    p = rd.power()
    if floor_normalise:
        db = 10 * np.log10(p / np.median(p) + 1e-30)
        vmin, vmax = 0.0, dyn_range_db
    else:
        db = rd.power_db(reference)
        vmin, vmax = -dyn_range_db, 0.0
    extent = [rd.doppler_hz[0], rd.doppler_hz[-1], rd.bistatic_range_m[0] / 1e3, rd.bistatic_range_m[-1] / 1e3]
    im = ax.imshow(db, extent=extent, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap,
                   interpolation="nearest")
    ax.set_xlabel("Doppler [Hz]")
    ax.set_ylabel("Bistatic range [km]")
    if title:
        ax.set_title(title, fontsize=9)
    for t in truth or []:
        ax.plot(t.doppler_hz, t.bistatic_range_m / 1e3, "o", mfc="none", mec="red", ms=12, mew=1.4)
    for d in detections or []:
        ax.plot(d.doppler_hz, d.bistatic_range_m / 1e3, "+", color="white", ms=8, mew=1.2)
    return im


def plot_cuts(ax_delay, ax_doppler, rd: RDMap, label: str, reference: float | None = None):
    db = rd.power_db(reference)
    i0, j0 = rd.cell(0.0, 0.0)
    ax_delay.plot(rd.bistatic_range_m / 1e3, db[:, j0], lw=0.8, label=label)
    ax_doppler.plot(rd.doppler_hz, db[i0, :], lw=0.8, label=label)
    ax_delay.set_xlabel("Bistatic range [km]  (zero-Doppler cut)")
    ax_doppler.set_xlabel("Doppler [Hz]  (zero-delay cut)")
    for ax in (ax_delay, ax_doppler):
        ax.set_ylabel("dB")
        ax.grid(alpha=0.3)
