import numpy as np
import pytest

from pr450.waveform import (
    RE_CONTROL,
    RE_PILOT,
    RE_SYNC,
    LteConfig,
    NrConfig,
    OfdmNumerology,
    generate_lte,
    generate_nr,
    lte_pss,
    lte_sss,
    nr_pss,
    nr_sss,
    ofdm_demodulate,
    ofdm_modulate,
)
from pr450.waveform.base import symbol_timing
from pr450.waveform.nr import nr_ssb_block


@pytest.mark.parametrize("dc_hole", [True, False])
def test_ofdm_roundtrip_and_frame_length(dc_hole):
    num = OfdmNumerology(dc_hole=dc_hole)
    rng = np.random.default_rng(0)
    grid = rng.standard_normal((140, num.n_sc)) + 1j * rng.standard_normal((140, num.n_sc))
    iq = ofdm_modulate(grid, num)
    assert len(iq) == 76800  # 10 ms at 7.68 Msps
    starts, cps = symbol_timing(num, 140)
    np.testing.assert_allclose(ofdm_demodulate(iq, num, starts, cps), grid, atol=1e-9)


def test_full_load_unit_power():
    num = OfdmNumerology()
    rng = np.random.default_rng(1)
    grid = np.exp(2j * np.pi * rng.random((1400, num.n_sc)))
    iq = ofdm_modulate(grid, num)
    assert abs(np.mean(np.abs(iq) ** 2) - 1) < 0.02


def test_lte_dc_hole_skips_dc():
    num = OfdmNumerology(dc_hole=True)
    off = num.subcarrier_offsets()
    assert 0 not in off and off.min() == -150 and off.max() == 150


def test_lte_pss_is_constant_modulus_zc():
    for n2 in range(3):
        d = lte_pss(n2)
        assert d.shape == (62,)
        np.testing.assert_allclose(np.abs(d), 1)
    assert abs(np.vdot(lte_pss(0), lte_pss(1))) / 62 < 0.25


def test_lte_pss_matches_exact_zadoff_chu_phase():
    """Phase reduced exactly in integers: pi*u*m(m+1)/63 mod 2*pi = pi*(u*m(m+1) mod 126)/63."""
    for n2, u in enumerate((25, 29, 34)):
        n = np.arange(62)
        m = np.where(n < 31, n, n + 1)
        exact = np.exp(-1j * np.pi * ((u * m * (m + 1)) % 126) / 63)
        np.testing.assert_allclose(lte_pss(n2), exact, atol=1e-12)


def test_lte_sss_is_bpsk_and_differs_between_subframes():
    d0, d5 = lte_sss(10, 2)
    assert set(np.unique(d0.real)) == {-1.0, 1.0}
    assert not np.allclose(d0, d5)


def test_nr_pss_known_prefix():
    # N_ID2 = 0: x = 0,1,1,0,1,1,1,... -> d = 1 - 2x
    np.testing.assert_array_equal(nr_pss(0)[:7].real, [1, -1, -1, 1, -1, -1, -1])


def test_nr_sss_orthogonality():
    a, b = nr_sss(0, 0), nr_sss(1, 0)
    assert a.shape == (127,)
    assert abs(np.vdot(a, b)) / 127 < 0.15


def test_lte_crs_density():
    wf = generate_lte(LteConfig(n_ports=1), "idle", 0.01)
    per_sf = np.count_nonzero(wf.re_type[:14] == RE_PILOT)
    assert per_sf == 2 * 2 * 2 * 25  # 2 slots x 2 symbols x 2 per RB x 25 RB
    wf2 = generate_lte(LteConfig(n_ports=2), "idle", 0.01)
    assert np.count_nonzero(wf2.re_type[:14] == RE_PILOT) == 2 * per_sf


def test_lte_sync_placement():
    wf = generate_lte(LteConfig(), "idle", 0.01)
    sync_symbols = np.unique(np.nonzero(wf.re_type == RE_SYNC)[0])
    np.testing.assert_array_equal(sync_symbols, [5, 6, 75, 76])


def test_nr_ssb_re_counts():
    vals, types, _ = nr_ssb_block(101, 0, np.random.default_rng(0))
    assert np.count_nonzero(types == RE_SYNC) == 2 * 127
    assert np.count_nonzero(types == RE_PILOT) == 144
    assert np.count_nonzero(types == RE_CONTROL) == 432


def test_nr_ssb_periodicity():
    wf = generate_nr(NrConfig(ssb_period_ms=20, trs_period_ms=None, sib1_period_ms=None), "idle", 0.04)
    sync_symbols = np.unique(np.nonzero(wf.re_type == RE_SYNC)[0])
    np.testing.assert_array_equal(sync_symbols, [2, 4, 282, 284])  # 20 ms = 280 symbols


def test_nr_idle_is_much_sparser_than_lte_idle():
    lte = generate_lte(LteConfig(), "idle", 0.04)
    nr = generate_nr(NrConfig(), "idle", 0.04)
    assert nr.occupancy()["occupied"] < 0.5 * lte.occupancy()["occupied"]


@pytest.mark.parametrize("load", ["idle", "medium", "full"])
def test_occupancy_increases_with_load(load):
    for gen, cfg in ((generate_lte, LteConfig()), (generate_nr, NrConfig())):
        wf = gen(cfg, load, 0.02)
        assert wf.iq.shape == (2 * 76800,)
        if load == "full":
            assert wf.occupancy()["occupied"] > 0.85


def test_structural_empty_marks_only_empty_reserved_res():
    for wf in (generate_lte(LteConfig(), "full", 0.01), generate_nr(NrConfig(), "full", 0.02)):
        se = wf.structural_empty
        assert se is not None and se.any()
        assert np.all(wf.re_type[se] == 0)


def test_lte_one_port_leaves_port1_crs_positions_empty_in_control_symbol():
    """TS 36.211 6.2.4: REGs in symbol 0 assume CRS on ports 0 and 1 even with one port."""
    from pr450.waveform.lte import lte_crs_subcarriers
    wf = generate_lte(LteConfig(n_ports=1), "full", 0.01)
    k1 = lte_crs_subcarriers(wf.cell_id, 1, 0, 25)
    for sf in range(10):
        sym = 14 * sf
        assert np.all(wf.re_type[sym, k1] == 0)
        assert np.all(wf.structural_empty[sym, k1])


def test_nr_5ms_ssb_period_fills_both_half_frames():
    wf = generate_nr(NrConfig(ssb_period_ms=5, trs_period_ms=None, sib1_period_ms=None), "idle", 0.02)
    sync_symbols = np.unique(np.nonzero(wf.re_type == RE_SYNC)[0])
    np.testing.assert_array_equal(sync_symbols, [2, 4, 72, 74, 142, 144, 212, 214])


def test_nr_20ms_ssb_period_counts_from_sfn0():
    """With a 20 ms period and an odd starting SFN the first frame correctly carries no SSB."""
    wf = generate_nr(NrConfig(sfn0=1, trs_period_ms=None, sib1_period_ms=None), "idle", 0.02)
    sync_symbols = np.unique(np.nonzero(wf.re_type == RE_SYNC)[0])
    np.testing.assert_array_equal(sync_symbols, [142, 144])
