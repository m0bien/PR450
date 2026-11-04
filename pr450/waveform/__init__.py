from .base import (
    KNOWN_TYPES,
    RE_CONTROL,
    RE_DATA,
    RE_EMPTY,
    RE_PILOT,
    RE_SYNC,
    RE_TYPE_NAMES,
    OfdmNumerology,
    Waveform,
    gold_sequence,
    ofdm_demodulate,
    ofdm_modulate,
    qam_hard_decision,
    random_qam,
)
from .load import LOAD_PRESETS, LoadProfile, get_load
from .lte import LteConfig, generate_lte, lte_pss, lte_sss
from .nr import NrConfig, generate_nr, nr_pbch_dmrs, nr_pss, nr_sss


def generate(cfg, load, duration_s: float, seed: int = 0) -> Waveform:
    """Dispatch to the LTE or NR generator based on the config type."""
    if isinstance(cfg, LteConfig):
        return generate_lte(cfg, load, duration_s, seed)
    if isinstance(cfg, NrConfig):
        return generate_nr(cfg, load, duration_s, seed)
    raise TypeError(f"unsupported config type {type(cfg).__name__}")


__all__ = [
    "KNOWN_TYPES", "RE_CONTROL", "RE_DATA", "RE_EMPTY", "RE_PILOT", "RE_SYNC", "RE_TYPE_NAMES",
    "OfdmNumerology", "Waveform", "gold_sequence", "ofdm_demodulate", "ofdm_modulate",
    "qam_hard_decision", "random_qam",
    "LOAD_PRESETS", "LoadProfile", "get_load",
    "LteConfig", "generate_lte", "lte_pss", "lte_sss",
    "NrConfig", "generate_nr", "nr_pbch_dmrs", "nr_pss", "nr_sss",
    "generate",
]
