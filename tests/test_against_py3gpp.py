"""
Cross-check the bit-exact NR parts of pr450 against py3gpp, an independent
open-source Python port of the MATLAB 5G Toolbox (itself tested against MATLAB).

py3gpp covers PSS/SSS, PBCH DMRS, the Gold sequence, PDSCH DMRS and OFDM
modulation. It has no CSI-RS (TRS) or PDCCH DMRS generator, and no LTE, so
those parts are not cross-checked here.

Skipped automatically when py3gpp is not installed (pip install py3gpp).
py3gpp is AGPL-3.0: keep it a test-only dependency.
"""
import numpy as np
import pytest

py3gpp = pytest.importorskip("py3gpp")

from pr450.waveform import RE_PILOT, RE_SYNC, NrConfig, generate_nr, gold_sequence  # noqa: E402
from pr450.waveform.base import OfdmNumerology, ofdm_modulate  # noqa: E402
from pr450.waveform.nr import SSB_SC, nr_pbch_dmrs, nr_pdsch_dmrs, nr_pss, nr_ssb_block, nr_sss  # noqa: E402

CELL_IDS = [0, 1, 2, 3, 101, 335, 503, 1007]


def _ssb_reference(cell_id: int, i_ssb: int) -> np.ndarray:
    """240 x 4 SSB grid (subcarrier x symbol) with PSS, SSS and PBCH DMRS from py3gpp."""
    ssb = np.zeros(SSB_SC * 4, dtype=complex)
    ssb[np.asarray(py3gpp.nrPSSIndices())] = py3gpp.nrPSS(cell_id)
    ssb[np.asarray(py3gpp.nrSSSIndices())] = py3gpp.nrSSS(cell_id)
    ssb[np.asarray(py3gpp.nrPBCHDMRSIndices(cell_id))] = py3gpp.nrPBCHDMRS(cell_id, i_ssb)
    return ssb.reshape(4, SSB_SC)  # 0-based column-major index k + 240 l -> row l


@pytest.mark.parametrize("c_init", [0, 1, 12345, 2**20 + 3, 2**30 + 7, 2**31 - 1])
def test_gold_sequence(c_init):
    ref = np.asarray(py3gpp.nrPRBS(c_init, 1000)).astype(int).ravel()
    np.testing.assert_array_equal(gold_sequence(c_init, 1000), ref)


@pytest.mark.parametrize("cell_id", CELL_IDS)
def test_pss_sss(cell_id):
    np.testing.assert_allclose(nr_pss(cell_id % 3), np.asarray(py3gpp.nrPSS(cell_id)), atol=1e-12)
    np.testing.assert_allclose(nr_sss(cell_id // 3, cell_id % 3), np.asarray(py3gpp.nrSSS(cell_id)), atol=1e-12)


@pytest.mark.parametrize("cell_id", CELL_IDS)
@pytest.mark.parametrize("i_ssb", range(4))
def test_pbch_dmrs_sequence(cell_id, i_ssb):
    np.testing.assert_allclose(nr_pbch_dmrs(cell_id, i_ssb), np.asarray(py3gpp.nrPBCHDMRS(cell_id, i_ssb)), atol=1e-12)


@pytest.mark.parametrize("cell_id", CELL_IDS)
@pytest.mark.parametrize("i_ssb", range(4))
def test_ssb_block_known_res(cell_id, i_ssb):
    """Positions *and* values of every known RE in the SSB."""
    vals, types, _ = nr_ssb_block(cell_id, i_ssb, np.random.default_rng(0))
    mine = np.where(np.isin(types, (RE_PILOT, RE_SYNC)), vals, 0)
    np.testing.assert_allclose(mine, _ssb_reference(cell_id, i_ssb), atol=1e-12)


@pytest.mark.parametrize("cell_id", [1, 101, 1007])
def test_ssb_in_generated_grid(cell_id):
    cfg = NrConfig(cell_id=cell_id, ssb_bitmap=(1, 1, 1, 1), trs_period_ms=None)
    wf = generate_nr(cfg, "idle", 0.002)
    known = np.where(np.isin(wf.re_type, (RE_PILOT, RE_SYNC)), wf.grid, 0)
    for i_ssb, start in enumerate((2, 8, 16, 22)):
        block = known[start:start + 4, cfg.ssb_k0:cfg.ssb_k0 + SSB_SC]
        np.testing.assert_allclose(block, _ssb_reference(cell_id, i_ssb), atol=1e-12)


def _pdsch_dmrs_reference(cell_id: int, slot: int, n_rb: int) -> np.ndarray:
    carrier = py3gpp.nrCarrierConfig(NCellID=cell_id, NSizeGrid=n_rb, NSlot=slot)
    pdsch = py3gpp.nrPDSCHConfig()
    pdsch.NSizeBWP = n_rb
    pdsch.NStartBWP = 0
    pdsch.PRBSet = list(range(n_rb))
    pdsch.SymbolAllocation = [2, 12]
    pdsch.DMRS.DMRSTypeAPosition = 2
    pdsch.DMRS.DMRSAdditionalPosition = 1
    pdsch.DMRS.DMRSLength = 1
    pdsch.DMRS.NIDNSCID = cell_id
    return np.asarray(py3gpp.nrPDSCHDMRS(pdsch, carrier)).reshape(2, -1)  # symbols 2 and 11


@pytest.mark.parametrize("cell_id", [1, 101, 1007])
@pytest.mark.parametrize("slot", [0, 3, 9])
def test_pdsch_dmrs_sequence(cell_id, slot):
    ref = _pdsch_dmrs_reference(cell_id, slot, 25)
    for row, l in enumerate((2, 11)):
        np.testing.assert_allclose(nr_pdsch_dmrs(cell_id, slot, l, 300), ref[row], atol=1e-12)


@pytest.mark.parametrize("cell_id", [1, 1007])
def test_pdsch_dmrs_in_generated_grid(cell_id):
    """Full-load slot without SSB/TRS/SIB1: DMRS on every even subcarrier of symbols 2 and 11."""
    cfg = NrConfig(cell_id=cell_id, trs_period_ms=None, sib1_period_ms=None)
    wf = generate_nr(cfg, "full", 0.004)
    slot = 3
    ref = _pdsch_dmrs_reference(cell_id, slot, 25)
    for row, l in enumerate((2, 11)):
        sym = 14 * slot + l
        pilots = np.flatnonzero(wf.re_type[sym] == RE_PILOT)
        np.testing.assert_array_equal(pilots, np.arange(0, 300, 2))
        np.testing.assert_allclose(wf.grid[sym, pilots], ref[row], atol=1e-12)


def test_ofdm_modulation_matches_up_to_scale():
    """Same CP lengths, subcarrier-to-frequency mapping and symbol timing as nrOFDMModulate."""
    num = OfdmNumerology(n_rb=25, dc_hole=False)
    rng = np.random.default_rng(5)
    grid = (rng.standard_normal((14, num.n_sc)) + 1j * rng.standard_normal((14, num.n_sc))) / np.sqrt(2)
    mine = ofdm_modulate(grid, num)
    carrier = py3gpp.nrCarrierConfig(NSizeGrid=25, SubcarrierSpacing=15)
    waveform, _info = py3gpp.nrOFDMModulate(carrier, grid.T, Nfft=512, SampleRate=7_680_000)
    ref = np.asarray(waveform).ravel()
    assert ref.shape == mine.shape
    scale = np.vdot(mine, ref) / np.vdot(mine, mine)
    err = np.linalg.norm(ref - scale * mine) / np.linalg.norm(ref)
    assert err < 1e-9, f"relative error {err:.2e} (scale {scale:.4g})"


@pytest.mark.parametrize("cell_id", [1, 1007])
def test_ssb_second_half_frame_uses_nhf_in_pbch_dmrs(cell_id):
    """5 ms SSB period: the burst in the second half-frame uses i_bar = i_SSB + 4 (38.211 7.4.1.4.1)."""
    cfg = NrConfig(cell_id=cell_id, ssb_period_ms=5, ssb_bitmap=(1, 0, 0, 0), trs_period_ms=None, sib1_period_ms=None)
    wf = generate_nr(cfg, "idle", 0.01)
    known = np.where(np.isin(wf.re_type, (RE_PILOT, RE_SYNC)), wf.grid, 0)
    block = known[70 + 2:70 + 6, cfg.ssb_k0:cfg.ssb_k0 + SSB_SC]     # slot 5, symbols 2..5
    ref = np.zeros(SSB_SC * 4, dtype=complex)
    ref[np.asarray(py3gpp.nrPSSIndices())] = py3gpp.nrPSS(cell_id)
    ref[np.asarray(py3gpp.nrSSSIndices())] = py3gpp.nrSSS(cell_id)
    ref[np.asarray(py3gpp.nrPBCHDMRSIndices(cell_id))] = py3gpp.nrPBCHDMRS(cell_id, 4)
    np.testing.assert_allclose(block, ref.reshape(4, SSB_SC), atol=1e-12)
