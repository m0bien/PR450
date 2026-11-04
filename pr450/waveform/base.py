"""
Shared OFDM machinery for the LTE and NR downlink generators.

LTE (5 MHz) and NR n72 (5 MHz, mu=0) share the same basic OFDM numbers:
15 kHz subcarrier spacing, 25 PRB = 300 subcarriers, FFT size 512, 7.68 Msps,
and the same normal cyclic prefix pattern (one long CP followed by six short
CPs every 0.5 ms). The only difference at this level is that LTE leaves the
DC subcarrier empty while NR does not.

Grid conventions used throughout the package:
  grid[symbol, k]  complex resource grid, k = 0..n_sc-1 in ascending frequency
  re_type[...]     int8 code for what each resource element carries (RE_*)
  mod_bits[...]    bits/symbol of unknown-payload REs (2=QPSK ... 8=256QAM),
                   0 for known or empty REs

The time-domain signal is scaled so that a fully loaded carrier (every RE
used, unit energy) has unit average sample power. A lightly loaded carrier
therefore has proportionally lower power -- exactly the effect the passive
radar must cope with.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

RE_EMPTY = 0
RE_PILOT = 1    # reference signals known a priori (CRS, DMRS, TRS, PBCH-DMRS)
RE_SYNC = 2     # PSS / SSS
RE_CONTROL = 3  # PBCH / PDCCH / PCFICH / PHICH payload (unknown bits)
RE_DATA = 4     # PDSCH payload (unknown bits)

KNOWN_TYPES = (RE_PILOT, RE_SYNC)
RE_TYPE_NAMES = {RE_EMPTY: "empty", RE_PILOT: "pilot", RE_SYNC: "sync", RE_CONTROL: "control", RE_DATA: "data"}


@dataclass(frozen=True)
class OfdmNumerology:
    """OFDM parameters for a normal-CP, 15 kHz carrier."""

    n_rb: int = 25
    scs_hz: float = 15e3
    nfft: int = 512
    dc_hole: bool = False

    @property
    def n_sc(self) -> int:
        return 12 * self.n_rb

    @property
    def fs(self) -> float:
        return self.nfft * self.scs_hz

    @property
    def cp_long(self) -> int:
        return 160 * self.nfft // 2048

    @property
    def cp_short(self) -> int:
        return 144 * self.nfft // 2048

    @property
    def symbols_per_ms(self) -> int:
        return 14

    def cp_lengths(self, n_sym: int) -> np.ndarray:
        idx = np.arange(n_sym)
        return np.where(idx % 7 == 0, self.cp_long, self.cp_short)

    def subcarrier_offsets(self) -> np.ndarray:
        """Frequency offset of each grid subcarrier, in units of scs, relative to the carrier centre."""
        k = np.arange(self.n_sc)
        half = self.n_sc // 2
        if self.dc_hole:
            return np.where(k < half, k - half, k - half + 1)
        return k - half

    def subcarrier_bins(self) -> np.ndarray:
        return self.subcarrier_offsets() % self.nfft


# ---------------------------------------------------------------------------
# Sequences and constellations
# ---------------------------------------------------------------------------

_NC = 1600
_X1_LEN = 1 << 15


def _x1_sequence(length: int) -> list[int]:
    x1 = [0] * length
    x1[0] = 1
    for n in range(length - 31):
        x1[n + 31] = x1[n + 3] ^ x1[n]
    return x1


_X1 = _x1_sequence(_X1_LEN)


@lru_cache(maxsize=8192)
def _gold(c_init: int, length: int) -> np.ndarray:
    total = _NC + length
    if total + 31 > _X1_LEN:
        raise ValueError("gold sequence too long")
    x2 = [(c_init >> i) & 1 for i in range(31)] + [0] * (total)
    for n in range(total):
        x2[n + 31] = x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n]
    c = np.array([_X1[n + _NC] ^ x2[n + _NC] for n in range(length)], dtype=np.uint8)
    c.setflags(write=False)
    return c


def gold_sequence(c_init: int, length: int) -> np.ndarray:
    """Length-31 Gold pseudo-random sequence c(n) of 36.211 7.2 / 38.211 5.2.1."""
    return _gold(int(c_init), int(length))


def qpsk_from_bits(c: np.ndarray) -> np.ndarray:
    """r(m) = ((1-2c(2m)) + j(1-2c(2m+1))) / sqrt(2)."""
    c = np.asarray(c, dtype=float)
    return ((1 - 2 * c[0::2]) + 1j * (1 - 2 * c[1::2])) / np.sqrt(2)


def _qam_levels(bits: int) -> tuple[np.ndarray, float]:
    m = 1 << (bits // 2)
    levels = np.arange(-(m - 1), m, 2, dtype=float)
    norm = np.sqrt(2 * (m * m - 1) / 3)
    return levels, norm


def random_qam(rng: np.random.Generator, bits: int, n: int) -> np.ndarray:
    """Unit-average-energy square QAM symbols with random content (bits = 2, 4, 6 or 8)."""
    levels, norm = _qam_levels(bits)
    i = rng.integers(0, len(levels), n)
    q = rng.integers(0, len(levels), n)
    return (levels[i] + 1j * levels[q]) / norm


def qam_hard_decision(x: np.ndarray, bits: int) -> np.ndarray:
    """Nearest square-QAM point (unit average energy) for each sample of x."""
    levels, norm = _qam_levels(bits)
    m = len(levels)

    def slice_axis(v):
        idx = np.clip(np.round((v * norm + (m - 1)) / 2), 0, m - 1).astype(int)
        return levels[idx]

    return (slice_axis(x.real) + 1j * slice_axis(x.imag)) / norm


# ---------------------------------------------------------------------------
# OFDM modulation / demodulation
# ---------------------------------------------------------------------------

def symbol_timing(num: OfdmNumerology, n_sym: int) -> tuple[np.ndarray, np.ndarray]:
    cps = num.cp_lengths(n_sym)
    starts = np.concatenate(([0], np.cumsum(cps + num.nfft)[:-1])).astype(np.int64)
    return starts, cps


def ofdm_modulate(grid: np.ndarray, num: OfdmNumerology) -> np.ndarray:
    n_sym = grid.shape[0]
    spectrum = np.zeros((n_sym, num.nfft), dtype=complex)
    spectrum[:, num.subcarrier_bins()] = grid
    body = np.fft.ifft(spectrum, axis=1) * (num.nfft / np.sqrt(num.n_sc))

    starts, cps = symbol_timing(num, n_sym)
    out = np.empty(int(starts[-1] + cps[-1] + num.nfft), dtype=complex)
    for cp in np.unique(cps):
        sel = np.flatnonzero(cps == cp)
        with_cp = np.concatenate([body[sel, num.nfft - cp:], body[sel]], axis=1)
        positions = starts[sel, None] + np.arange(cp + num.nfft)[None, :]
        out[positions] = with_cp
    return out


def ofdm_demodulate(
    iq: np.ndarray,
    num: OfdmNumerology,
    starts: np.ndarray,
    cps: np.ndarray,
    sample_offset: int = 0,
) -> np.ndarray:
    """FFT each symbol (window placed right after its CP) and return the n_sym x n_sc grid."""
    idx = starts + cps + sample_offset
    rows = iq[idx[:, None] + np.arange(num.nfft)[None, :]]
    spectrum = np.fft.fft(rows, axis=1)
    return spectrum[:, num.subcarrier_bins()] * (np.sqrt(num.n_sc) / num.nfft)


# ---------------------------------------------------------------------------
# Waveform container
# ---------------------------------------------------------------------------

@dataclass
class Waveform:
    standard: str
    num: OfdmNumerology
    grid: np.ndarray
    re_type: np.ndarray
    mod_bits: np.ndarray
    iq: np.ndarray
    sym_starts: np.ndarray
    cp_lens: np.ndarray
    cell_id: int
    load_name: str
    meta: dict = field(default_factory=dict)
    structural_empty: np.ndarray | None = None  # REs the standard leaves empty (e.g. SSB / PSS guard REs)

    @property
    def fs(self) -> float:
        return self.num.fs

    @property
    def n_symbols(self) -> int:
        return self.grid.shape[0]

    @property
    def duration_s(self) -> float:
        return len(self.iq) / self.fs

    @property
    def fc_hz(self) -> float:
        return float(self.meta.get("fc_hz", 463.5e6))

    def known_mask(self) -> np.ndarray:
        return np.isin(self.re_type, KNOWN_TYPES)

    def synthesize(self, grid: np.ndarray | None = None, mask: np.ndarray | None = None) -> np.ndarray:
        """Time-domain signal from `grid` (default: the transmitted grid), optionally keeping only `mask` REs."""
        g = self.grid if grid is None else grid
        if mask is not None:
            g = np.where(mask, g, 0)
        return ofdm_modulate(g, self.num)

    def occupancy(self) -> dict:
        total = self.re_type.size
        out = {name: float(np.count_nonzero(self.re_type == code)) / total for code, name in RE_TYPE_NAMES.items()}
        out["occupied"] = 1.0 - out["empty"]
        return out


def build_waveform(standard, num, grid, re_type, mod_bits, cell_id, load_name, meta, structural_empty=None) -> Waveform:
    starts, cps = symbol_timing(num, grid.shape[0])
    return Waveform(
        standard=standard,
        num=num,
        grid=grid,
        re_type=re_type,
        mod_bits=mod_bits,
        iq=ofdm_modulate(grid, num),
        sym_starts=starts,
        cp_lens=cps,
        cell_id=cell_id,
        load_name=load_name,
        meta=meta,
        structural_empty=structural_empty,
    )


def fill_random(
    grid: np.ndarray,
    re_type: np.ndarray,
    mod_bits: np.ndarray,
    mask: np.ndarray,
    kind: int,
    bits: int,
    rng: np.random.Generator,
) -> None:
    """Fill the REs selected by `mask` (views into the full arrays) with random QAM payload."""
    n = int(np.count_nonzero(mask))
    if n == 0:
        return
    grid[mask] = random_qam(rng, bits, n)
    re_type[mask] = kind
    mod_bits[mask] = bits
