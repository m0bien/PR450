"""
Bistatic geometry and link budget.

All positions are metres in a local Cartesian frame (x east, y north, z up).
Bistatic range is the excess path length relative to the direct path:

    R_b = |p - tx| + |p - rx| - |rx - tx|

so the direct-path signal sits at R_b = 0 and delay tau = R_b / c.
Bistatic Doppler:

    f_d = -(1/lambda) * d/dt (|p - tx| + |p - rx|)
        = -(1/lambda) * (u_tx + u_rx) . v,   u_x = (p - x) / |p - x|
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C = 299_792_458.0
K_BOLTZMANN = 1.380649e-23


@dataclass
class Target:
    name: str
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    rcs_dbsm: float


@dataclass
class InjectedEcho:
    """A synthetic echo placed directly by bistatic range, Doppler and SNR.

    snr_db is per sample relative to receiver noise for a fully loaded carrier
    (the same definition as TargetTruth.snr_db), so test targets can be put
    exactly where a Monte Carlo study needs them, independent of geometry.
    """
    name: str
    bistatic_range_m: float
    doppler_hz: float
    snr_db: float


@dataclass
class ClutterSpec:
    n_scatterers: int = 40
    max_radius_m: float = 2000.0
    rcs_dbsm_mean: float = 30.0
    rcs_dbsm_std: float = 8.0
    seed: int = 1


@dataclass
class Scenario:
    tx_position_m: tuple[float, float, float] = (0.0, 0.0, 30.0)
    rx_position_m: tuple[float, float, float] = (3000.0, 0.0, 10.0)
    eirp_dbm: float = 61.0
    ref_gain_dbi: float = 10.0
    surv_gain_dbi: float = 10.0
    dpi_suppression_db: float = 25.0
    ref_multipath: list[tuple[float, float]] = field(default_factory=lambda: [(150.0, -15.0), (420.0, -22.0)])
    noise_figure_db: float = 5.0
    temperature_k: float = 290.0
    targets: list[Target] = field(default_factory=list)
    injected: list[InjectedEcho] = field(default_factory=list)
    clutter: ClutterSpec = field(default_factory=ClutterSpec)
    adc_bits: int | None = 12
    adc_backoff_db: float = 12.0

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(np.subtract(self.rx_position_m, self.tx_position_m)))


def bistatic_range_doppler(tx, rx, pos, vel, wavelength_m: float) -> tuple[float, float, float, float]:
    """Return (bistatic range m, Doppler Hz, |p - tx|, |p - rx|)."""
    tx, rx, pos, vel = (np.asarray(a, dtype=float) for a in (tx, rx, pos, vel))
    d_tx = pos - tx
    d_rx = pos - rx
    r_tx = float(np.linalg.norm(d_tx))
    r_rx = float(np.linalg.norm(d_rx))
    baseline = float(np.linalg.norm(rx - tx))
    rate = float(np.dot(d_tx / r_tx + d_rx / r_rx, vel))
    return r_tx + r_rx - baseline, -rate / wavelength_m, r_tx, r_rx


def noise_power_dbm(sample_rate_hz: float, noise_figure_db: float, temperature_k: float = 290.0) -> float:
    return 10 * np.log10(K_BOLTZMANN * temperature_k * sample_rate_hz * 1e3) + noise_figure_db


def friis_dbm(eirp_dbm: float, gain_dbi: float, distance_m: float, wavelength_m: float) -> float:
    return eirp_dbm + gain_dbi + 20 * np.log10(wavelength_m / (4 * np.pi * distance_m))


def bistatic_echo_dbm(eirp_dbm, gain_dbi, rcs_dbsm, r_tx, r_rx, wavelength_m) -> float:
    return (
        eirp_dbm + gain_dbi + rcs_dbsm + 20 * np.log10(wavelength_m)
        - 30 * np.log10(4 * np.pi) - 20 * np.log10(r_tx) - 20 * np.log10(r_rx)
    )


def random_clutter(spec: ClutterSpec, rx) -> list[tuple[np.ndarray, float]]:
    """Static point scatterers on the ground within max_radius_m of the receiver."""
    rng = np.random.default_rng(spec.seed)
    radius = spec.max_radius_m * np.sqrt(rng.random(spec.n_scatterers))
    angle = rng.uniform(0, 2 * np.pi, spec.n_scatterers)
    rcs = rng.normal(spec.rcs_dbsm_mean, spec.rcs_dbsm_std, spec.n_scatterers)
    rx = np.asarray(rx, dtype=float)
    pts = []
    for r, a, s in zip(radius, angle, rcs):
        pos = np.array([rx[0] + max(r, 50.0) * np.cos(a), rx[1] + max(r, 50.0) * np.sin(a), 0.0])
        pts.append((pos, float(s)))
    return pts
