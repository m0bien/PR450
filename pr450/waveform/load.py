"""
Cell-load profiles shared by the LTE and NR generators.

Real traffic is bursty: at low load most TTIs carry no user data at all,
and the TTIs that do are often only partly allocated. A profile therefore
has two knobs: the probability that a 1 ms TTI schedules user data, and
the fraction of PRBs allocated in an active TTI. Mean PRB utilisation is
their product.

"full" and "full_64qam" stand in for the 3GPP conformance test models
(E-TM1.1 / NR-FR1-TM1.1 with QPSK, and E-TM3.1 / NR-FR1-TM3.1 with 64QAM).
They match the TMs in occupancy and modulation, not in the exact
per-PRB power boosting, which hardly matters for the ambiguity function.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadProfile:
    name: str
    tti_activity: float
    prb_fraction: float
    data_bits: int = 2

    @property
    def mean_prb_utilisation(self) -> float:
        return self.tti_activity * self.prb_fraction


LOAD_PRESETS: dict[str, LoadProfile] = {
    "idle": LoadProfile("idle", 0.0, 0.0, 2),
    "low": LoadProfile("low", 0.3, 0.33, 4),
    "medium": LoadProfile("medium", 0.6, 0.5, 6),
    "high": LoadProfile("high", 0.9, 0.8, 6),
    "full": LoadProfile("full", 1.0, 1.0, 2),
    "full_64qam": LoadProfile("full_64qam", 1.0, 1.0, 6),
}


def get_load(load: str | LoadProfile) -> LoadProfile:
    if isinstance(load, LoadProfile):
        return load
    try:
        return LOAD_PRESETS[load]
    except KeyError:
        raise ValueError(f"unknown load preset {load!r}; choose from {sorted(LOAD_PRESETS)}") from None
