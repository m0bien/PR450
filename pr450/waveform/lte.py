"""
LTE FDD downlink generator (TS 36.211), aimed at Band 72 / Band 31 at 5 MHz.

What is modelled exactly (bit-true to 36.211):
  - PSS (Zadoff-Chu, 6.11.1) and SSS (m-sequences, 6.11.2) in subframes 0/5
  - Cell-specific reference signals, CRS (6.10.1) for 1 or 2 antenna ports

What is modelled statistically (right REs and modulation, random content):
  - PBCH in subframe 0, slot 1, symbols 0..3 of the central 72 subcarriers
  - Control region (PCFICH/PHICH/PDCCH) in the first `cfi` symbols
  - PDSCH, including a small SIB1 allocation in subframe 5 of even frames

Multiple antenna ports are summed into one grid, which is equivalent to
assuming the same propagation channel from every port. This is adequate for
a single-receiver passive radar study.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import (
    RE_CONTROL,
    RE_DATA,
    RE_EMPTY,
    RE_PILOT,
    RE_SYNC,
    OfdmNumerology,
    Waveform,
    build_waveform,
    fill_random,
    gold_sequence,
    qpsk_from_bits,
)
from .load import LoadProfile, get_load

N_RB_MAX = 110


@dataclass
class LteConfig:
    n_rb: int = 25
    cell_id: int = 1
    n_ports: int = 1
    cfi: int = 2
    sib1: bool = True
    sfn0: int = 0
    fc_hz: float = 463.5e6
    band: str = "B72"


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

def lte_pss(n_id2: int) -> np.ndarray:
    u = (25, 29, 34)[n_id2]
    n = np.arange(62)
    nn = np.where(n < 31, n, n + 1)
    return np.exp(-1j * np.pi * u * nn * (nn + 1) / 63)


def _mseq31(taps: tuple[int, ...]) -> np.ndarray:
    x = [0, 0, 0, 0, 1] + [0] * 26
    for i in range(26):
        x[i + 5] = sum(x[i + t] for t in taps) % 2
    return 1 - 2 * np.array(x)


def lte_sss(n_id1: int, n_id2: int) -> tuple[np.ndarray, np.ndarray]:
    """SSS for subframe 0 and subframe 5."""
    s_t = _mseq31((2, 0))
    c_t = _mseq31((3, 0))
    z_t = _mseq31((4, 2, 1, 0))

    qp = n_id1 // 30
    q = (n_id1 + qp * (qp + 1) // 2) // 30
    mp = n_id1 + q * (q + 1) // 2
    m0 = mp % 31
    m1 = (m0 + mp // 31 + 1) % 31

    n = np.arange(31)
    s0, s1 = s_t[(n + m0) % 31], s_t[(n + m1) % 31]
    c0, c1 = c_t[(n + n_id2) % 31], c_t[(n + n_id2 + 3) % 31]
    z_m0, z_m1 = z_t[(n + m0 % 8) % 31], z_t[(n + m1 % 8) % 31]

    d0 = np.empty(62)
    d5 = np.empty(62)
    d0[0::2], d0[1::2] = s0 * c0, s1 * c1 * z_m0
    d5[0::2], d5[1::2] = s1 * c0, s0 * c1 * z_m1
    return d0.astype(complex), d5.astype(complex)


def lte_crs_values(cell_id: int, ns: int, l: int, n_rb: int) -> np.ndarray:
    c_init = (1 << 10) * (7 * (ns + 1) + l + 1) * (2 * cell_id + 1) + 2 * cell_id + 1
    r = qpsk_from_bits(gold_sequence(c_init, 4 * N_RB_MAX))
    return r[np.arange(2 * n_rb) + N_RB_MAX - n_rb]


def lte_crs_subcarriers(cell_id: int, port: int, l: int, n_rb: int) -> np.ndarray:
    if port == 0:
        v = 0 if l == 0 else 3
    elif port == 1:
        v = 3 if l == 0 else 0
    else:
        raise ValueError("only antenna ports 0 and 1 are supported")
    return 6 * np.arange(2 * n_rb) + (v + cell_id % 6) % 6


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_lte(
    cfg: LteConfig,
    load: str | LoadProfile,
    duration_s: float,
    seed: int = 0,
) -> Waveform:
    load = get_load(load)
    num = OfdmNumerology(n_rb=cfg.n_rb, dc_hole=True)
    rng = np.random.default_rng(seed)

    n_sf = int(round(duration_s * 1000))
    n_sc = num.n_sc
    grid = np.zeros((14 * n_sf, n_sc), dtype=complex)
    re_type = np.zeros(grid.shape, dtype=np.int8)
    mod_bits = np.zeros(grid.shape, dtype=np.int8)
    structural_empty = np.zeros(grid.shape, dtype=bool)

    cell = cfg.cell_id
    n_id1, n_id2 = cell // 3, cell % 3
    vshift = cell % 6
    pss = lte_pss(n_id2)
    sss0, sss5 = lte_sss(n_id1, n_id2)

    center = n_sc // 2
    sync_k = np.arange(62) - 31 + center
    central = np.zeros(n_sc, dtype=bool)
    central[np.arange(72) - 36 + center] = True
    k_all = np.arange(n_sc)
    pbch_crs_exclusion = (k_all - vshift) % 3 == 0

    for sf_abs in range(n_sf):
        sfn = cfg.sfn0 + sf_abs // 10
        sf = sf_abs % 10
        base = 14 * sf_abs
        g = grid[base:base + 14]
        t = re_type[base:base + 14]
        b = mod_bits[base:base + 14]
        reserved = np.zeros((14, n_sc), dtype=bool)

        # CRS
        for slot in (0, 1):
            ns = 2 * sf + slot
            for l in (0, 4):
                vals = lte_crs_values(cell, ns, l, cfg.n_rb)
                for port in range(cfg.n_ports):
                    k = lte_crs_subcarriers(cell, port, l, cfg.n_rb)
                    g[7 * slot + l, k] = vals
                    t[7 * slot + l, k] = RE_PILOT

        # PSS / SSS: last two symbols of slot 0 in subframes 0 and 5
        if sf in (0, 5):
            g[6, sync_k] = pss
            g[5, sync_k] = sss0 if sf == 0 else sss5
            t[6, sync_k] = RE_SYNC
            t[5, sync_k] = RE_SYNC
            reserved[5:7, central] = True

        # PBCH: subframe 0, slot 1, symbols 0..3, central 72 subcarriers
        if sf == 0:
            for l in range(4):
                sym = 7 + l
                mask = central & (t[sym] == RE_EMPTY)
                if l < 2:
                    mask &= ~pbch_crs_exclusion
                row_mask = np.zeros((14, n_sc), dtype=bool)
                row_mask[sym] = mask
                fill_random(g, t, b, row_mask, RE_CONTROL, 2, rng)
                reserved[sym, central] = True

        active = rng.random() < load.tti_activity
        sib1_now = cfg.sib1 and sf == 5 and sfn % 2 == 0

        # Guard REs around PSS/SSS and REs PBCH leaves for CRS ports that are not transmitted
        structural_empty[base:base + 14] = reserved & (t == RE_EMPTY)

        # Control region, occupied in PRB x symbol blocks (a coarse stand-in for REG clustering)
        ctrl_fraction = min(1.0, 0.1 + 0.5 * active + 0.15 * sib1_now)
        ctrl = np.zeros((14, n_sc), dtype=bool)
        used_blocks = rng.random((cfg.cfi, cfg.n_rb)) < ctrl_fraction
        ctrl[:cfg.cfi] = np.repeat(used_blocks, 12, axis=1) & (t[:cfg.cfi] == RE_EMPTY)
        fill_random(g, t, b, ctrl, RE_CONTROL, 2, rng)
        reserved[:cfg.cfi] = True

        # PDSCH
        free_prbs = rng.permutation(cfg.n_rb)
        allocations = []
        if sib1_now:
            allocations.append((free_prbs[:4], 2))
            free_prbs = free_prbs[4:]
        if active:
            n_alloc = min(len(free_prbs), max(1, int(round(load.prb_fraction * cfg.n_rb))))
            allocations.append((free_prbs[:n_alloc], load.data_bits))
        for prbs, bits in allocations:
            kmask = np.zeros(n_sc, dtype=bool)
            for p in prbs:
                kmask[12 * p:12 * p + 12] = True
            mask = (t == RE_EMPTY) & ~reserved & kmask[None, :]
            fill_random(g, t, b, mask, RE_DATA, bits, rng)

    meta = {"fc_hz": cfg.fc_hz, "band": cfg.band, "n_ports": cfg.n_ports, "cfi": cfg.cfi, "sfn0": cfg.sfn0}
    return build_waveform("LTE", num, grid, re_type, mod_bits, cell, load.name, meta, structural_empty)
