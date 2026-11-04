import numpy as np
import pytest

from pr450.core import caf_batched, caf_ofdm, detect, eca, eca_b
from pr450.reference import demod_remod, pilot_only
from pr450.scenario import (
    C,
    ClutterSpec,
    Scenario,
    Target,
    apply_fir,
    bistatic_range_doppler,
    fractional_delay_fir,
    simulate,
)
from pr450.waveform import LteConfig, NrConfig, generate_lte, generate_nr


@pytest.fixture(scope="module")
def lte_wf():
    return generate_lte(LteConfig(), "full", 0.05, seed=3)


def _echo(s, delay, doppler, gain, fs):
    t = np.arange(len(s)) / fs
    return apply_fir(s, fractional_delay_fir([delay], [gain])) * np.exp(2j * np.pi * doppler * t)


def test_bistatic_doppler_matches_numerical_derivative():
    tx, rx = np.array([0, 0, 30.0]), np.array([3000, 0, 10.0])
    p, v = np.array([4000, 2500, 2.0]), np.array([-15, 10, 0.0])
    lam = C / 463.5e6
    rb, fd, _, _ = bistatic_range_doppler(tx, rx, p, v, lam)
    dt = 1e-3
    rb2, *_ = bistatic_range_doppler(tx, rx, p + v * dt, v, lam)
    assert fd == pytest.approx(-(rb2 - rb) / dt / lam, rel=1e-4)


def test_eca_removes_dpi_and_clutter(lte_wf):
    s = lte_wf.iq
    surv = apply_fir(s, fractional_delay_fir([0, 7.3, 20, 41.6], [1000, 300j, -120, 50]))
    for res in (eca(surv, s, 64), eca_b(surv, s, 64, batch_len=8192)):
        assert res.cancellation_db > 60


def test_caf_batched_locates_target(lte_wf):
    s, fs = lte_wf.iq, lte_wf.fs
    surv = _echo(s, 100.0, 80.0, 1.0, fs)
    rd = caf_batched(surv, s, fs, 200, doppler_max_hz=300)
    i, j = np.unravel_index(np.argmax(rd.power()), rd.data.shape)
    assert i == 100
    assert abs(rd.doppler_hz[j] - 80.0) <= abs(rd.doppler_hz[1] - rd.doppler_hz[0])


def test_caf_ofdm_locates_target_within_cp(lte_wf):
    s, fs = lte_wf.iq, lte_wf.fs
    surv = _echo(s, 20.0, -50.0, 1.0, fs)
    rd = caf_ofdm(surv, lte_wf, lte_wf.grid, 64, doppler_max_hz=300)
    i, j = np.unravel_index(np.argmax(rd.power()), rd.data.shape)
    assert i == 20
    assert abs(rd.doppler_hz[j] + 50.0) <= abs(rd.doppler_hz[1] - rd.doppler_hz[0])


def test_cfar_detects_target_in_noise(lte_wf):
    s, fs = lte_wf.iq, lte_wf.fs
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s))) / np.sqrt(2)
    surv = _echo(s, 150.0, 40.0, 0.05, fs) + noise  # -26 dB per sample, +~50 dB after integration
    rd = caf_batched(surv, s, fs, 256, doppler_max_hz=200)
    dets = detect(rd, pfa=1e-6)
    assert dets and dets[0].delay_idx == 150


@pytest.mark.parametrize("make", [lambda: generate_lte(LteConfig(), "medium", 0.02, seed=1),
                                  lambda: generate_nr(NrConfig(), "medium", 0.02, seed=1)])
def test_demod_remod_is_exact_at_high_snr(make):
    wf = make()
    rng = np.random.default_rng(0)
    rx = 30.0 * np.exp(0.7j) * wf.iq + (rng.standard_normal(len(wf.iq)) + 1j * rng.standard_normal(len(wf.iq))) * 0.01
    iq_hat, grid_hat, stats = demod_remod(rx, wf)
    assert stats.symbol_error_rate == 0
    assert stats.missed_re_fraction == 0 and stats.false_re_fraction == 0
    np.testing.assert_allclose(iq_hat, wf.iq, atol=1e-9)


def test_pilot_only_keeps_only_known_energy():
    wf = generate_nr(NrConfig(), "full", 0.02)
    p = pilot_only(wf)
    assert np.mean(np.abs(p) ** 2) < 0.2 * np.mean(np.abs(wf.iq) ** 2)


def test_simulation_truth_and_levels():
    wf = generate_lte(LteConfig(), "full", 0.02)
    sc = Scenario(
        targets=[Target("t", (4000, 2500, 2), (-15, 10, 0), 10)],
        clutter=ClutterSpec(n_scatterers=5),
        adc_bits=None,
    )
    sim = simulate(wf, sc)
    assert len(sim.ref) == len(sim.surv) == len(wf.iq)
    assert sim.levels_db["ref_dnr_db"] > sim.levels_db["surv_dnr_db"]
    assert sim.truth[0].bistatic_range_m == pytest.approx(4409.6, abs=1)


def test_eca_needs_precursor_taps_for_subsample_clutter(lte_wf):
    s = lte_wf.iq
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s))) / np.sqrt(2)
    surv = apply_fir(s, fractional_delay_fir([0.0, 3.4, 9.7], [3000, 800j, -400])) + noise
    without = 10 * np.log10(np.mean(np.abs(eca(surv, s, 48).output) ** 2))
    with_lead = 10 * np.log10(np.mean(np.abs(eca(surv, s, 48, n_lead=16).output) ** 2))
    assert with_lead < 0.5 < without  # residual at thermal noise only with pre-cursor taps


@pytest.mark.parametrize("std", ["lte", "nr"])
@pytest.mark.parametrize("load", ["idle", "full_64qam"])
def test_remod_reference_is_exact_through_full_receiver_model(std, load):
    """Through multipath, noise and a 12-bit ADC the remod grid must be exact:
    even -40 dB of reference error would leave ~+10 dB of uncancelled DPI."""
    gen, cfg = (generate_lte, LteConfig(n_ports=2)) if std == "lte" else (generate_nr, NrConfig())
    wf = gen(cfg, load, 0.05, seed=2)
    sim = simulate(wf, Scenario(clutter=ClutterSpec(n_scatterers=5)), seed=2)
    _, grid_hat, stats = demod_remod(sim.ref, wf)
    assert stats.false_re_fraction == 0 and stats.missed_re_fraction == 0
    np.testing.assert_allclose(grid_hat, wf.grid, atol=1e-9)


def test_range_taper_suppresses_near_in_range_sidelobes(lte_wf):
    """The occupied spectrum is sharp-edged (300 of 512 bins): without a taper the
    range response has a slowly decaying sinc-like ridge next to every target."""
    s, fs = lte_wf.iq, lte_wf.fs
    surv = _echo(s, 100.0, 40.0, 1.0, fs)

    def near_ridge_db(**kw):
        rd = caf_batched(surv, s, fs, 256, doppler_max_hz=100, window="blackmanharris", **kw)
        p = rd.power()
        i, j = np.unravel_index(np.argmax(p), p.shape)
        near = np.r_[p[i - 30:i - 4, j], p[i + 5:i + 31, j]]
        return 10 * np.log10(near.max() / p[i, j])

    assert near_ridge_db(range_window="blackmanharris", occupied_bw_hz=300 * 15e3) < near_ridge_db() - 10


def test_energy_normalisation_flattens_load_modulation():
    """LTE never goes silent (CRS), so per-batch energy equalisation removes the
    amplitude modulation that bursty traffic imposes on every echo."""
    wf = generate_lte(LteConfig(), "low", 0.1, seed=4)
    s, fs = wf.iq, wf.fs
    surv = _echo(s, 80.0, 30.0, 1.0, fs)

    def doppler_ridge_db(alpha):
        rd = caf_batched(surv, s, fs, 128, doppler_max_hz=400, window="blackmanharris", normalize=alpha)
        p = rd.power()
        i, j = np.unravel_index(np.argmax(p), p.shape)
        off = np.abs(rd.doppler_hz - rd.doppler_hz[j]) > 20
        return 10 * np.log10(np.median(p[i, off]) / p[i, j])

    assert doppler_ridge_db(1.0) < doppler_ridge_db(0.0) - 6


def test_clean_removes_strong_target_and_reveals_weak_one(lte_wf):
    from pr450.core import EcaCanceller, clean
    s, fs = lte_wf.iq, lte_wf.fs
    rng = np.random.default_rng(3)
    noise = (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s))) / np.sqrt(2)
    strong = _echo(s, 120.3, 25.4, 3.0, fs)
    weak = _echo(s, 131.7, 25.0 + 37.0, 0.02, fs)           # ~ -40 dB below the strong one
    surv = 50.0 * s + strong + weak + noise                 # plus a direct path for the canceller
    caf_kw = dict(doppler_max_hz=200, window="blackmanharris", range_window="blackmanharris", occupied_bw_hz=300 * 15e3)
    canceller = EcaCanceller(s, 16, n_lead=4)
    res = clean(surv, s, fs, 256, caf_kw, dict(pfa=1e-6, min_delay_samples=2, exclude_zero_doppler_hz=3.0),
                threshold_db=18.0, canceller=lambda v: canceller(v).output)
    c0 = res.components[0]
    assert abs(c0.delay_samples - 120.3) < 0.1 and abs(c0.doppler_hz - 25.4) < 0.3
    assert c0.residual_drop_db > 30
    found = [d for d in res.detections if abs(d.delay_s * fs - 131.7) < 2 and abs(d.doppler_hz - 62.0) < 3]
    assert found, "weak target not detected after CLEAN"


def test_injected_echo_lands_at_requested_range_doppler_and_snr():
    from pr450.scenario import InjectedEcho
    from pr450.core import fit_echo
    wf = generate_lte(LteConfig(), "full", 0.05, seed=6)
    sc = Scenario(injected=[InjectedEcho("x", 6000.0, -37.0, -10.0)], clutter=ClutterSpec(n_scatterers=0),
                  dpi_suppression_db=200.0, adc_bits=None)
    sim = simulate(wf, sc, seed=1)
    t = sim.truth[0]
    assert t.delay_samples == pytest.approx(6000.0 / C * wf.fs)
    echo = fit_echo(sim.surv, wf.iq, wf.fs, t.delay_samples, -37.0, poly_order=0)
    p = np.mean(np.abs(echo) ** 2) / np.mean(np.abs(wf.iq) ** 2)
    assert 10 * np.log10(p) == pytest.approx(-10.0, abs=0.1)


@pytest.mark.parametrize("seed", [7, 8])
def test_remod_exact_at_idle_with_12bit_adc(seed):
    """Regression: with a 12-bit ADC at idle load, quantisation put spurious energy
    (35-47 x pilot noise) into empty REs; 2-4 false REs per CPI raised the ECA
    residual from 0.9 to 3.3 dB above thermal. Fixed by an absolute energy floor."""
    wf = generate_lte(LteConfig(n_ports=2), "idle", 0.5, seed=seed)
    sim = simulate(wf, Scenario(), seed=seed)
    _, grid_hat, _ = demod_remod(sim.ref, wf)
    assert int(np.sum(np.abs(grid_hat - wf.grid) > 1e-9)) == 0
