"""
YAML configuration loading.

A waveform config has `standard: lte` or `standard: nr`; every other key maps
onto LteConfig / NrConfig. A scenario config maps onto Scenario, with
`targets` as a list and `clutter` as a mapping.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import yaml

from .scenario.geometry import ClutterSpec, Scenario, Target
from .waveform import LteConfig, NrConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def _read(path_or_dict) -> dict:
    if isinstance(path_or_dict, dict):
        return dict(path_or_dict)
    path = Path(path_or_dict)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build(cls, d: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


def load_waveform_config(path_or_dict) -> LteConfig | NrConfig:
    d = _read(path_or_dict)
    standard = str(d.pop("standard", "")).lower()
    if standard == "lte":
        return _build(LteConfig, d)
    if standard == "nr":
        if "ssb_bitmap" in d:
            d["ssb_bitmap"] = tuple(d["ssb_bitmap"])
        return _build(NrConfig, d)
    raise ValueError("waveform config needs `standard: lte` or `standard: nr`")


def load_scenario(path_or_dict) -> Scenario:
    d = _read(path_or_dict)
    d["targets"] = [_build(Target, {**t, "position_m": tuple(t["position_m"]), "velocity_mps": tuple(t["velocity_mps"])})
                    for t in d.get("targets", [])]
    if "clutter" in d:
        d["clutter"] = _build(ClutterSpec, d["clutter"])
    if "ref_multipath" in d:
        d["ref_multipath"] = [tuple(x) for x in d["ref_multipath"]]
    for key in ("tx_position_m", "rx_position_m"):
        if key in d:
            d[key] = tuple(d[key])
    return _build(Scenario, d)
