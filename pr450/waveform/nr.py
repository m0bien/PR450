"""
NR FDD downlink generator (TS 38.211), aimed at n72 / n31 at 5 MHz, mu = 0.

What is modelled exactly (bit-true to 38.211):
  - PSS / SSS (7.4.2.2, 7.4.2.3) and PBCH DMRS (7.4.1.4) inside the SSB
  - PDSCH DMRS, type 1, single symbol (7.4.1.1)
  - PDCCH DMRS (7.4.1.3), CSI-RS sequence for the TRS (7.4.1.5)

What is modelled statistically (right REs and modulation, random content):
  - PBCH payload, PDCCH payload, PDSCH payload (incl. a periodic SIB1)

Simplifications, which a later revision can replace with the real procedure:
  - SSB placed at a configurable subcarrier offset (sync raster ignored)
  - SSB Case A, L = 4 candidate positions at symbols {2, 8, 16, 22} of the
    first half-frame; n_hf = 0
  - CORESET REG selection is random (no CCE-to-REG interleaving)
  - DMRS/TRS sequence indices count from the carrier's first subcarrier
    (point A assumed at the carrier edge)
  - PDSCH is not scheduled in SSB PRBs during SSB slots

The key point for the radar: NR has no always-on reference signal. An idle
cell transmits only SSB bursts (every ssb_period_ms), periodic SIB1 and TRS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
    random_qam,
)
from .load import LoadProfile, get_load

SSB_SC = 240
SSB_START_SYMBOLS = (2, 8, 16, 22)  # Case A, f <= 3 GHz, L = 4


@dataclass
class NrConfig:
    n_rb: int = 25
    cell_id: int = 1
    ssb_period_ms: int = 20
    ssb_bitmap: tuple[int, ...] = field(default=(1, 0, 0, 0))
    ssb_k0: int = 30
    coreset_symbols: int = 2
    coreset_rb_start: int = 0
    coreset_n_rb: int = 24
    pdcch_fraction: float = 0.5
    dmrs_additional_pos: int = 1
    trs_period_ms: int | None = 20
    trs_slot_offset: int = 4
    trs_k0: int = 0
    sib1_period_ms: int | None = 20
    sib1_slot: int = 2
    sib1_n_rb: int = 4
    sfn0: int = 0
    fc_hz: float = 463.5e6
    band: str = "n72"


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

def _mseq127(taps: tuple[int, ...], init: list[int]) -> np.ndarray:
    x = list(init) + [0] * 120
    for i in range(120):
        x[i + 7] = sum(x[i + t] for t in taps) % 2
    return np.array(x)


def nr_pss(n_id2: int) -> np.ndarray:
    x = _mseq127((4, 0), [0, 1, 1, 0, 1, 1, 1])
    n = np.arange(127)
    return (1 - 2 * x[(n + 43 * n_id2) % 127]).astype(complex)


def nr_sss(n_id1: int, n_id2: int) -> np.ndarray:
    x0 = _mseq127((4, 0), [1, 0, 0, 0, 0, 0, 0])
    x1 = _mseq127((1, 0), [1, 0, 0, 0, 0, 0, 0])
    m0 = 15 * (n_id1 // 112) + 5 * n_id2
    m1 = n_id1 % 112
    n = np.arange(127)
    return ((1 - 2 * x0[(n + m0) % 127]) * (1 - 2 * x1[(n + m1) % 127])).astype(complex)


def nr_pbch_dmrs(cell_id: int, i_ssb_bar: int) -> np.ndarray:
    c_init = (1 << 11) * (i_ssb_bar + 1) * (cell_id // 4 + 1) + (1 << 6) * (i_ssb_bar + 1) + cell_id % 4
    return qpsk_from_bits(gold_sequence(c_init, 288))


def _dmrs_c_init(slot: int, l: int, n_id: int) -> int:
    return ((1 << 17) * (14 * slot + l + 1) * (2 * n_id + 1) + 2 * n_id) % (1 << 31)


def nr_pdsch_dmrs(cell_id: int, slot: int, l: int, n_sc: int) -> np.ndarray:
    """DMRS values for every even subcarrier of the carrier (index k//2)."""
    return qpsk_from_bits(gold_sequence(_dmrs_c_init(slot, l, cell_id), n_sc))


def nr_pdcch_dmrs(cell_id: int, slot: int, l: int, n_rb: int) -> np.ndarray:
    """DMRS values for subcarriers 12p+1, 12p+5, 12p+9 (index 3p+j)."""
    return qpsk_from_bits(gold_sequence(_dmrs_c_init(slot, l, cell_id), 6 * n_rb))


def nr_csirs(cell_id: int, slot: int, l: int, n_values: int) -> np.ndarray:
    c_init = ((1 << 10) * (14 * slot + l + 1) * (2 * cell_id + 1) + cell_id) % (1 << 31)
    return qpsk_from_bits(gold_sequence(c_init, 2 * n_values))


def nr_ssb_block(cell_id: int, i_ssb: int, rng: np.random.Generator, n_hf: int = 0):
    """4 x 240 SSB: values, RE types and modulation bits."""
    vals = np.zeros((4, SSB_SC), dtype=complex)
    types = np.zeros((4, SSB_SC), dtype=np.int8)
    bits = np.zeros((4, SSB_SC), dtype=np.int8)

    k_sync = np.arange(56, 183)
    vals[0, k_sync] = nr_pss(cell_id % 3)
    vals[2, k_sync] = nr_sss(cell_id // 3, cell_id % 3)
    types[0, k_sync] = RE_SYNC
    types[2, k_sync] = RE_SYNC

    nu = cell_id % 4
    k = np.arange(SSB_SC)
    edge = (k < 48) | (k >= 192)
    pbch_region = {1: np.ones(SSB_SC, bool), 2: edge, 3: np.ones(SSB_SC, bool)}
    dmrs = nr_pbch_dmrs(cell_id, i_ssb + 4 * n_hf)   # L = 4: i_bar = i_SSB + 4 n_hf (38.211 7.4.1.4.1)
    pos = 0
    for sym in (1, 2, 3):
        region = pbch_region[sym]
        dk = k[region & (k % 4 == nu)]
        vals[sym, dk] = dmrs[pos:pos + len(dk)]
        types[sym, dk] = RE_PILOT
        pos += len(dk)
        data_k = k[region & (k % 4 != nu)]
        vals[sym, data_k] = random_qam(rng, 2, len(data_k))
        types[sym, data_k] = RE_CONTROL
        bits[sym, data_k] = 2
    return vals, types, bits


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_nr(
    cfg: NrConfig,
    load: str | LoadProfile,
    duration_s: float,
    seed: int = 0,
) -> Waveform:
    load = get_load(load)
    num = OfdmNumerology(n_rb=cfg.n_rb, dc_hole=False)
    n_sc = num.n_sc
    if cfg.ssb_k0 + SSB_SC > n_sc:
        raise ValueError("SSB does not fit in the carrier at this ssb_k0")
    if cfg.coreset_rb_start + cfg.coreset_n_rb > cfg.n_rb:
        raise ValueError("CORESET does not fit in the carrier")
    rng = np.random.default_rng(seed)

    n_slots = int(round(duration_s * 1000))
    grid = np.zeros((14 * n_slots, n_sc), dtype=complex)
    re_type = np.zeros(grid.shape, dtype=np.int8)
    mod_bits = np.zeros(grid.shape, dtype=np.int8)
    structural_empty = np.zeros(grid.shape, dtype=bool)

    cell = cfg.cell_id
    cs = cfg.coreset_symbols
    dmrs_symbols = [2 if cs <= 2 else 3]
    if cfg.dmrs_additional_pos >= 1:
        dmrs_symbols.append(11)
    ssb_prbs = np.arange(cfg.ssb_k0 // 12, (cfg.ssb_k0 + SSB_SC - 1) // 12 + 1)
    coreset_prbs = np.arange(cfg.coreset_rb_start, cfg.coreset_rb_start + cfg.coreset_n_rb)
    even_k = np.arange(0, n_sc, 2)

    for slot_abs in range(n_slots):
        frame = cfg.sfn0 + slot_abs // 10
        ns = slot_abs % 10
        t_ms = frame * 10 + ns
        base = 14 * slot_abs
        g = grid[base:base + 14]
        t = re_type[base:base + 14]
        b = mod_bits[base:base + 14]
        reserved = np.zeros((14, n_sc), dtype=bool)
        blocked_prbs: set[int] = set()

        # SSB burst in each half-frame that starts on an ssb_period_ms boundary counted from
        # SFN 0 (so with a 20 ms period and an odd sfn0 the first frame correctly has none);
        # a 5 ms period puts a burst in both half-frames. Case A: slots 0-1 of the half-frame.
        hf, rel = divmod(ns, 5)
        if (frame * 10 + 5 * hf) % cfg.ssb_period_ms == 0 and rel in (0, 1):
            for i_ssb, start in enumerate(SSB_START_SYMBOLS):
                if not cfg.ssb_bitmap[i_ssb] or start // 14 != rel:
                    continue
                s0 = start % 14
                vals, types, bits = nr_ssb_block(cell, i_ssb, rng, n_hf=hf)
                ks = slice(cfg.ssb_k0, cfg.ssb_k0 + SSB_SC)
                g[s0:s0 + 4, ks] = vals
                t[s0:s0 + 4, ks] = types
                b[s0:s0 + 4, ks] = bits
                reserved[s0:s0 + 4, ks] = True
                structural_empty[base + s0:base + s0 + 4, ks] = types == RE_EMPTY
                blocked_prbs.update(ssb_prbs.tolist())

        # TRS: two consecutive slots, symbols 4 and 8, every 4th subcarrier
        if cfg.trs_period_ms and (t_ms - cfg.trs_slot_offset) % cfg.trs_period_ms in (0, 1):
            k_trs = np.arange(cfg.trs_k0, n_sc, 4)
            for l in (4, 8):
                vals = nr_csirs(cell, ns, l, len(k_trs))
                free = t[l, k_trs] == RE_EMPTY
                g[l, k_trs[free]] = vals[free]
                t[l, k_trs[free]] = RE_PILOT

        active = rng.random() < load.tti_activity
        sib1_now = bool(cfg.sib1_period_ms) and (t_ms - cfg.sib1_slot) % cfg.sib1_period_ms == 0

        # PDCCH in the CORESET
        if active or sib1_now:
            fraction = cfg.pdcch_fraction if active else 0.25
            for l in range(cs):
                dmrs = nr_pdcch_dmrs(cell, ns, l, cfg.n_rb)
                for p in coreset_prbs[rng.random(len(coreset_prbs)) < fraction]:
                    k = 12 * p + np.arange(12)
                    if np.any(t[l, k] != RE_EMPTY):
                        continue
                    dk = 12 * p + np.array([1, 5, 9])
                    g[l, dk] = dmrs[3 * p + np.arange(3)]
                    t[l, dk] = RE_PILOT
                    data_k = np.setdiff1d(k, dk)
                    g[l, data_k] = random_qam(rng, 2, len(data_k))
                    t[l, data_k] = RE_CONTROL
                    b[l, data_k] = 2
        reserved[:cs] = True

        # PDSCH allocations (SIB1 first, then user data)
        free_prbs = np.array([p for p in rng.permutation(cfg.n_rb) if p not in blocked_prbs], dtype=int)
        allocations = []
        if sib1_now and len(free_prbs):
            allocations.append((free_prbs[:cfg.sib1_n_rb], 2))
            free_prbs = free_prbs[cfg.sib1_n_rb:]
        if active and len(free_prbs):
            n_alloc = min(len(free_prbs), max(1, int(round(load.prb_fraction * cfg.n_rb))))
            allocations.append((free_prbs[:n_alloc], load.data_bits))

        for prbs, bits in allocations:
            kmask = np.zeros(n_sc, dtype=bool)
            for p in prbs:
                kmask[12 * p:12 * p + 12] = True
            for l in dmrs_symbols:
                dk = even_k[kmask[even_k] & (t[l, even_k] == RE_EMPTY)]
                g[l, dk] = nr_pdsch_dmrs(cell, ns, l, n_sc)[dk // 2]
                t[l, dk] = RE_PILOT
            mask = (t == RE_EMPTY) & ~reserved & kmask[None, :]
            fill_random(g, t, b, mask, RE_DATA, bits, rng)

    meta = {
        "fc_hz": cfg.fc_hz,
        "band": cfg.band,
        "ssb_period_ms": cfg.ssb_period_ms,
        "ssb_bitmap": tuple(cfg.ssb_bitmap),
        "trs_period_ms": cfg.trs_period_ms,
        "sfn0": cfg.sfn0,
    }
    return build_waveform("NR", num, grid, re_type, mod_bits, cell, load.name, meta, structural_empty)
