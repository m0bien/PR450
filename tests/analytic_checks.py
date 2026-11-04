"""
Analytic self-checks for every pr450 stage.

Each check compares the code against something known *independently* of the
code: a 3GPP-specified number, a physical law, a closed-form statistical
result, or a mathematical invariant of the algorithm. They are plain
functions (no pytest fixtures) so that tests/test_mutations.py can re-run
them against deliberately broken code and prove that they catch the bug.

`rec(name, measured, expected)` records values for the self-test report.
"""
from __future__ import annotations

import numpy as np

from pr450.core import EcaCanceller, caf_batched, caf_ofdm, clean, fit_echo
from pr450.core.cfar import ca_cfar
from pr450.reference import demod_remod, estimate_static_channel, pilot_only
from pr450.scenario import (
    C,
    ClutterSpec,
    Scenario,
    Target,
    apply_fir,
    bistatic_echo_dbm,
    bistatic_range_doppler,
    fractional_delay_fir,
    friis_dbm,
    noise_power_dbm,
    quantize,
    simulate,
)
from pr450.waveform import LteConfig, NrConfig, OfdmNumerology, generate_lte, generate_nr, ofdm_demodulate, ofdm_modulate

FC = 463.5e6
LAM = C / FC
BW = 300 * 15e3


def _noop(*_args, **_kw):
    pass


def _noise(n, rng):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


def _db(x):
    return 10 * np.log10(x)


# =============================================================================
# Waveform generation
# =============================================================================

def check_subcarrier_frequency_mapping(rec=_noop):
    """A single active subcarrier must come out as a tone at offset * 15 kHz
    (LTE skips DC, NR does not). A mirrored or shifted spectrum survives the
    modulate/demodulate round trip, so it needs this absolute check."""
    for dc_hole, k, offset in ((True, 150, +1), (True, 149, -1), (False, 150, 0), (False, 160, +10), (True, 0, -150)):
        num = OfdmNumerology(dc_hole=dc_hole)
        grid = np.zeros((14, num.n_sc), dtype=complex)
        grid[:, k] = 1
        iq = ofdm_modulate(grid, num)
        body = iq[num.cp_long:num.cp_long + num.nfft]  # first symbol, CP removed
        f_meas = np.fft.fftfreq(num.nfft, 1 / num.fs)[np.argmax(np.abs(np.fft.fft(body)))]
        rec(f"tone k={k} {'LTE' if dc_hole else 'NR'} [kHz]", f_meas / 1e3, offset * 15.0)
        assert abs(f_meas - offset * 15e3) < 1.0, f"subcarrier {k} at {f_meas:.0f} Hz, expected {offset * 15e3:.0f} Hz"


def check_frame_timing_per_spec(rec=_noop):
    """Sample positions from TS 36.211 / 38.211 arithmetic (7.68 Msps, normal CP):
    a frame is 76800 samples; slot-0 symbol l starts at 40+512 + (l-1)*(36+512)."""
    def start(l):
        return 0 if l == 0 else 552 + (l - 1) * 548

    lte = generate_lte(LteConfig(), "idle", 0.01)
    nr = generate_nr(NrConfig(), "idle", 0.01)
    rec("samples per frame", len(lte.iq), 76800)
    assert len(lte.iq) == len(nr.iq) == 76800
    for l in range(7):
        assert lte.sym_starts[l] == start(l), f"symbol {l} starts at {lte.sym_starts[l]}, spec {start(l)}"
    assert lte.sym_starts[7] == 3840, "second slot must start at 0.5 ms"

    # PSS really sits there: correlate the record with the PSS-only symbol body
    for wf, sym in ((lte, 6), (nr, 2)):
        pss_only = np.where(wf.re_type[sym] == 2, wf.grid[sym], 0)
        body = ofdm_modulate(np.vstack([pss_only, np.zeros((13, wf.num.n_sc))]), wf.num)[40:40 + 512]
        corr = np.abs(np.correlate(wf.iq[:20000], body, mode="valid"))
        expected = start(sym) + 36
        rec(f"{wf.standard} PSS sample index", int(np.argmax(corr)), expected)
        assert np.argmax(corr) == expected


def check_load_power_consistency(rec=_noop):
    """Parseval: mean sample power equals occupied RE energy / n_sc (unit RE energy)."""
    for gen, cfg in ((generate_lte, LteConfig()), (generate_nr, NrConfig())):
        for load in ("idle", "medium", "full"):
            wf = gen(cfg, load, 0.02, seed=1)
            predicted = np.sum(np.abs(wf.grid) ** 2) / wf.n_symbols / wf.num.n_sc
            cp_factor = 1.0  # CP copies part of each symbol: average power unchanged for white OFDM
            measured = np.mean(np.abs(wf.iq) ** 2)
            rec(f"{wf.standard} {load} power [dB]", _db(measured), _db(predicted * cp_factor))
            assert abs(_db(measured) - _db(predicted)) < 0.15


# =============================================================================
# Receiver / scenario model
# =============================================================================

def check_thermal_noise_and_friis(rec=_noop):
    k_t0 = noise_power_dbm(1.0, 0.0, 290.0)
    rec("kT0 [dBm/Hz]", k_t0, -173.98)
    assert abs(k_t0 + 173.98) < 0.01
    fspl = -friis_dbm(0.0, 0.0, 1000.0, LAM)          # 32.45 + 20log10(f_MHz) + 20log10(d_km)
    expected = 32.45 + 20 * np.log10(463.5)
    rec("FSPL 1 km @ 463.5 MHz [dB]", fspl, expected)
    assert abs(fspl - expected) < 0.02


def check_radar_equation_monostatic_limit(rec=_noop):
    """Bistatic equation with R_tx = R_rx must reduce to Pt G^2 lambda^2 sigma / ((4 pi)^3 R^4)."""
    pt_w, g, sigma, r = 20.0, 10 ** 1.5, 3.0, 2500.0
    pr_w = pt_w * g * g * LAM**2 * sigma / ((4 * np.pi) ** 3 * r**4)
    expected_dbm = 10 * np.log10(pr_w * 1e3)
    got = bistatic_echo_dbm(10 * np.log10(pt_w * 1e3) + 15.0, 15.0, 10 * np.log10(sigma), r, r, LAM)
    rec("monostatic echo [dBm]", got, expected_dbm)
    assert abs(got - expected_dbm) < 1e-6


def check_bistatic_doppler_physics(rec=_noop):
    """A target closing on both Tx and Rx shortens the path -> positive Doppler,
    magnitude (|v_tx| + |v_rx|) / lambda for motion along both lines of sight."""
    tx, rx = np.array([0.0, 0.0, 0.0]), np.array([4000.0, 0.0, 0.0])
    pos = np.array([2000.0, 3000.0, 0.0])                # equidistant from Tx and Rx
    toward_baseline = np.array([0.0, -20.0, 0.0])
    rb, fd, r_tx, _ = bistatic_range_doppler(tx, rx, pos, toward_baseline, LAM)
    closing = 2 * 20.0 * 3000.0 / r_tx                    # each range shrinks at v * cos(angle)
    rec("closing target Doppler [Hz]", fd, closing / LAM)
    assert fd > 0 and abs(fd - closing / LAM) < 1e-6
    assert abs(rb - (2 * r_tx - 4000.0)) < 1e-6
    _, fd_away, _, _ = bistatic_range_doppler(tx, rx, pos, -toward_baseline, LAM)
    assert fd_away == -fd


def check_simulated_echo_matches_truth(rec=_noop):
    """The surveillance channel must contain each target at the truth delay,
    Doppler and link-budget power; receiver noise must have unit variance."""
    wf = generate_lte(LteConfig(), "full", 0.1, seed=2)
    sc = Scenario(targets=[Target("t", (5000.0, 3000.0, 50.0), (-12.0, 8.0, 0.0), 25.0)],
                  clutter=ClutterSpec(n_scatterers=0), dpi_suppression_db=200.0, adc_bits=None)
    sim = simulate(wf, sc, seed=3)
    t = sim.truth[0]
    echo = fit_echo(sim.surv, wf.iq, wf.fs, t.delay_samples, t.doppler_hz, half_taps=8, poly_order=0)
    p_echo = np.mean(np.abs(echo) ** 2) / np.mean(np.abs(wf.iq) ** 2)
    noise_var = np.mean(np.abs(sim.surv - echo) ** 2)
    rec("echo power vs link budget [dB]", _db(p_echo), t.snr_db)
    rec("receiver noise variance", noise_var, 1.0)
    assert abs(_db(p_echo) - t.snr_db) < 0.1
    assert abs(noise_var - 1.0) < 0.02

    # and it is found at the truth position by an independent correlation
    rd = caf_batched(sim.surv, wf.iq, wf.fs, 512, doppler_max_hz=300, window=None)
    i, j = np.unravel_index(np.argmax(rd.power()), rd.data.shape)
    df = rd.doppler_hz[1] - rd.doppler_hz[0]
    rec("echo delay [samples]", i, t.delay_samples)
    rec("echo Doppler [Hz]", rd.doppler_hz[j], t.doppler_hz)
    assert abs(i - t.delay_samples) <= 0.5 and abs(rd.doppler_hz[j] - t.doppler_hz) <= df


def check_adc_sqnr(rec=_noop):
    """Uniform quantiser on a Gaussian signal: SQNR = 6.02 b + 1.76 - backoff [dB] (no clipping)."""
    rng = np.random.default_rng(0)
    x = _noise(400_000, rng)
    for bits, backoff in ((8, 12.0), (12, 12.0)):
        q, clipped = quantize(x, bits, backoff)
        sqnr = _db(np.mean(np.abs(x) ** 2) / np.mean(np.abs(q - x) ** 2))
        expected = 6.02 * bits + 1.76 - backoff
        rec(f"SQNR {bits} bit [dB]", sqnr, expected)
        assert abs(sqnr - expected) < 0.5 and clipped < 1e-4


# =============================================================================
# Reference reconstruction
# =============================================================================

def check_channel_estimate_matches_true_channel(rec=_noop):
    """Estimate of a known integer-delay multipath channel vs its exact frequency response."""
    for gen, cfg in ((generate_lte, LteConfig()), (generate_nr, NrConfig())):
        wf = gen(cfg, "medium", 0.05, seed=5)
        delays, gains = np.array([0, 3, 11]), np.array([2.0, 0.5j, -0.25])
        rx = apply_fir(wf.iq, fractional_delay_fir(delays, gains))
        rx = rx + 1e-4 * _noise(len(rx), np.random.default_rng(1))
        y = ofdm_demodulate(rx, wf.num, wf.sym_starts, wf.cp_lens)
        h_est = estimate_static_channel(y, wf)
        off = wf.num.subcarrier_offsets()
        h_true = (gains[None, :] * np.exp(-2j * np.pi * np.outer(off, delays) / wf.num.nfft)).sum(axis=1)
        err_db = _db(np.sum(np.abs(h_est - h_true) ** 2) / np.sum(np.abs(h_true) ** 2))
        rec(f"{wf.standard} channel estimate error [dB]", err_db, "< -40")
        assert err_db < -40


def check_pilot_only_energy(rec=_noop):
    """Pilot-only reference energy must equal the known-RE share of the grid energy."""
    wf = generate_nr(NrConfig(), "medium", 0.04, seed=2)
    e = np.abs(wf.grid) ** 2
    expected = e[wf.known_mask()].sum() / e.sum()
    measured = np.sum(np.abs(pilot_only(wf)) ** 2) / np.sum(np.abs(wf.iq) ** 2)
    rec("pilot-only energy fraction", measured, expected)
    assert abs(measured / expected - 1) < 0.01


def check_remod_exact_under_snr(rec=_noop):
    """Through a realistic reference channel the remodulated grid must be bit-exact."""
    for gen, cfg in ((generate_lte, LteConfig(n_ports=2)), (generate_nr, NrConfig())):
        wf = gen(cfg, "medium", 0.05, seed=7)
        sim = simulate(wf, Scenario(clutter=ClutterSpec(n_scatterers=3)), seed=7)
        _, grid_hat, st = demod_remod(sim.ref, wf)
        wrong = int(np.count_nonzero(np.abs(grid_hat - wf.grid) > 1e-9))
        rec(f"{wf.standard} remod wrong REs", wrong, 0)
        assert wrong == 0


# =============================================================================
# ECA
# =============================================================================

def _shift(x, lag):
    """y[t] = x[t - lag], zero-filled (the ECA dictionary column for that lag)."""
    if lag >= 0:
        return np.concatenate([np.zeros(lag, dtype=x.dtype), x[:len(x) - lag]])
    return np.concatenate([x[-lag:], np.zeros(-lag, dtype=x.dtype)])


def _eca_case(seed=0, n=60_000):
    rng = np.random.default_rng(seed)
    ref = _noise(n, rng)
    x = apply_fir(ref, fractional_delay_fir([0.0, 2.6, 9.3, 17.0], [50.0, 10j, -4.0, 2.0])) + _noise(n, rng)
    return ref, x


def check_eca_orthogonality_and_idempotence(rec=_noop):
    """Least squares: the output is orthogonal to every dictionary column, and a
    second pass removes nothing more. Run without diagonal loading, which by
    design leaves a bias of diag_load * |w| (~5e-9 here) in the projection."""
    ref, x = _eca_case()
    taps, lead = 24, 6
    canc = EcaCanceller(ref, taps, n_lead=lead, diag_load=0.0)
    out = canc(x).output
    proj = np.array([np.vdot(_shift(ref, lag), out) for lag in range(-lead, taps)])
    rel = np.max(np.abs(proj)) / np.sqrt(np.sum(np.abs(out) ** 2) * np.sum(np.abs(ref) ** 2))
    rec("ECA residual / dictionary correlation", rel, "< 1e-12")
    assert rel < 1e-12, f"ECA output not orthogonal to its dictionary (rel {rel:.1e})"
    again = canc(out).output
    change = np.linalg.norm(again - out) / np.linalg.norm(out)
    rec("ECA idempotence change", change, "< 1e-9")
    assert change < 1e-9


def check_eca_fast_path_equals_generic(rec=_noop):
    ref, x = _eca_case(seed=1)
    fast = EcaCanceller(ref, 20, batch_len=8192, n_lead=4)(x).output
    generic = EcaCanceller(ref, 20, batch_len=8192, n_lead=4, window=10_000)(x).output
    diff = np.max(np.abs(fast - generic)) / np.max(np.abs(fast))
    rec("ECA fast vs generic", diff, "< 1e-9")
    assert diff < 1e-9


def check_eca_preserves_moving_target(rec=_noop):
    """Zero-Doppler cancellation must not touch a target well away from 0 Hz."""
    wf = generate_lte(LteConfig(), "full", 0.1, seed=3)
    s, fs = wf.iq, wf.fs
    t = np.arange(len(s)) / fs
    target = 0.05 * apply_fir(s, fractional_delay_fir([40.0], [1.0])) * np.exp(2j * np.pi * 60.0 * t)
    x = 100.0 * s + apply_fir(s, fractional_delay_fir([5.0], [10.0])) + target
    out = EcaCanceller(s, 32, n_lead=4)(x).output
    before = np.abs(np.vdot(target, target))
    after = np.abs(np.vdot(target, out))
    loss = _db(before / after)
    rec("moving target loss through ECA [dB]", loss, "< 0.1")
    assert loss < 0.1


# =============================================================================
# Range-Doppler map (CAF)
# =============================================================================

def _window_loss(w):
    w = np.asarray(w, dtype=float)
    return np.sum(w) ** 2 / (len(w) * np.sum(w**2))


def check_caf_processing_gain(rec=_noop):
    """Output SNR = |a|^2 * P_s * N * (window losses) for an on-grid target in
    unit-variance noise. Catches batch misalignment, wrong normalisation,
    wrong axes and window bugs anywhere in the CAF."""
    from scipy.signal import get_window
    wf = generate_lte(LteConfig(), "full", 0.1, seed=11)
    s, fs = wf.iq, wf.fs
    rng = np.random.default_rng(4)
    a, delay, batch = 0.03, 100, 512
    nb = (len(s) - 256) // batch
    fd = 7 * fs / (batch * nb)                               # exactly on a Doppler bin
    t = np.arange(len(s)) / fs
    echo = a * np.concatenate([np.zeros(delay), s[:-delay]]) * np.exp(2j * np.pi * fd * t)
    x = echo + _noise(len(s), rng)
    p_s = np.mean(np.abs(s) ** 2)
    for name, kw, loss in (
        ("rect", dict(window=None), 1.0),
        ("Blackman-Harris", dict(window="blackmanharris", range_window="blackmanharris", occupied_bw_hz=BW),
         _window_loss(get_window("blackmanharris", nb, fftbins=False))
         * _window_loss(get_window("blackmanharris", 300, fftbins=False))),
    ):
        rd = caf_batched(x, s, fs, 256, batch_len=batch, **kw)
        p = rd.power()
        j = int(np.argmin(np.abs(rd.doppler_hz - fd)))
        noise = np.mean(np.delete(p, np.s_[delay - 8:delay + 9], axis=0)[:, np.abs(rd.doppler_hz - fd) > 30])
        snr = _db(p[delay, j] / noise)
        expected = _db(a**2 * p_s * nb * batch * loss)
        rec(f"CAF output SNR {name} [dB]", snr, expected)
        assert abs(snr - expected) < 0.5, f"{name}: {snr:.2f} dB vs theory {expected:.2f} dB"
    rec("delay bin [s]", rd.delay_s[1] - rd.delay_s[0], 1 / fs)
    rec("Doppler bin [Hz]", rd.doppler_hz[1] - rd.doppler_hz[0], fs / (batch * nb))
    assert np.isclose(rd.delay_s[1] - rd.delay_s[0], 1 / fs)
    assert np.isclose(rd.doppler_hz[1] - rd.doppler_hz[0], fs / (batch * nb))


def check_caf_implementations_agree(rec=_noop):
    """Batched (time-domain) and OFDM-domain CAF must put a target in the same cell."""
    wf = generate_nr(NrConfig(), "full", 0.1, seed=12)
    s, fs = wf.iq, wf.fs
    t = np.arange(len(s)) / fs
    x = np.concatenate([np.zeros(25), s[:-25]]) * np.exp(2j * np.pi * -42.0 * t)
    a = caf_batched(x, s, fs, 64, doppler_max_hz=200)
    b = caf_ofdm(x, wf, wf.grid, 64, doppler_max_hz=200)
    ia, ja = np.unravel_index(np.argmax(a.power()), a.data.shape)
    ib, jb = np.unravel_index(np.argmax(b.power()), b.data.shape)
    rec("batched vs OFDM CAF delay", (ia, ib), "equal")
    rec("batched vs OFDM CAF Doppler [Hz]", (a.doppler_hz[ja], b.doppler_hz[jb]), "-42")
    assert ia == ib == 25
    assert abs(a.doppler_hz[ja] + 42) < 10 and abs(b.doppler_hz[jb] + 42) < 10


# =============================================================================
# Detection (CFAR)
# =============================================================================

_GUARD, _TRAIN = (2, 3), (4, 8)


def _n_train():
    gd, gf = _GUARD
    td, tf = _TRAIN
    return (2 * (gd + td) + 1) * (2 * (gf + tf) + 1) - (2 * gd + 1) * (2 * gf + 1)


def check_cfar_false_alarm_rate(rec=_noop):
    """Exponential (square-law Gaussian) noise: empirical Pfa must match nominal."""
    rng = np.random.default_rng(8)
    p = rng.exponential(1.0, (800, 800))
    pfa = 1e-3
    mask, _ = ca_cfar(p, _GUARD, _TRAIN, pfa)
    inner = mask[20:-20, 20:-20]
    measured = inner.mean()
    sigma = np.sqrt(pfa / inner.size)
    rec("CFAR Pfa", measured, pfa)
    assert abs(measured - pfa) < 5 * sigma + 0.1 * pfa, f"Pfa {measured:.2e} vs nominal {pfa:.0e}"


def check_cfar_detection_probability(rec=_noop):
    """Swerling-1 target in CA-CFAR: Pd = (1 + alpha / (N (1 + SNR)))^-N."""
    rng = np.random.default_rng(9)
    n = _n_train()
    pfa = 1e-4
    alpha = n * (pfa ** (-1 / n) - 1)
    p = rng.exponential(1.0, (1200, 1200))
    rows, cols = np.meshgrid(np.arange(30, 1170, 30), np.arange(30, 1170, 30), indexing="ij")
    for snr_db in (10.0, 15.0):
        q = p.copy()
        snr = 10 ** (snr_db / 10)
        q[rows, cols] = rng.exponential(1.0 + snr, rows.shape)
        mask, _ = ca_cfar(q, _GUARD, _TRAIN, pfa)
        pd = mask[rows, cols].mean()
        theory = (1 + alpha / (n * (1 + snr))) ** (-n)
        rec(f"CFAR Pd @ {snr_db:.0f} dB", pd, theory)
        assert abs(pd - theory) < 4 * np.sqrt(theory * (1 - theory) / rows.size) + 0.01


# =============================================================================
# CLEAN
# =============================================================================

def check_clean_noop_without_targets(rec=_noop):
    """No detection above threshold -> no components, residual = canceller output."""
    ref, x = _eca_case(seed=3, n=80_000)
    canc = EcaCanceller(ref, 24, n_lead=6)
    res = clean(x, ref, 7.68e6, 128, dict(doppler_max_hz=200), dict(pfa=1e-6, min_delay_samples=2,
                exclude_zero_doppler_hz=3.0), threshold_db=18.0, canceller=lambda v: canc(v).output)
    rec("CLEAN components on target-free data", len(res.components), 0)
    assert len(res.components) == 0
    assert np.allclose(res.residual, canc(x).output)


def check_clean_does_not_disturb_other_targets(rec=_noop):
    """After CLEAN removes a strong target, a weaker one's RD SNR must be what it
    is with the weak target alone (+-1 dB): the strong echo's self-noise is gone
    and the weak echo itself was not touched."""
    wf = generate_lte(LteConfig(), "full", 0.1, seed=13)
    s, fs = wf.iq, wf.fs
    t = np.arange(len(s)) / fs
    rng = np.random.default_rng(5)
    noise = _noise(len(s), rng)
    def echo(d, f, a):
        return a * apply_fir(s, fractional_delay_fir([d], [1.0])) * np.exp(2j * np.pi * f * t)
    weak = echo(190.4, -73.0, 0.01)
    caf_kw = dict(doppler_max_hz=200, window="blackmanharris", range_window="blackmanharris", occupied_bw_hz=BW)
    det_kw = dict(pfa=1e-6, min_delay_samples=2, exclude_zero_doppler_hz=3.0)

    def weak_snr(rd):
        p = rd.power()
        i0, j0 = rd.cell(190.4 / fs, -73.0)
        return _db(p[i0 - 1:i0 + 2, j0 - 1:j0 + 2].max() / np.median(p) * np.log(2))

    alone = caf_batched(weak + noise, s, fs, 256, **caf_kw)
    res = clean(weak + echo(60.7, 21.3, 2.0) + noise, s, fs, 256, caf_kw, det_kw, threshold_db=18.0)
    rec("weak target SNR alone / after CLEAN [dB]", (round(weak_snr(alone), 2), round(weak_snr(res.rd), 2)), "equal +-1")
    assert len(res.components) >= 1
    assert abs(weak_snr(alone) - weak_snr(res.rd)) < 1.0


# =============================================================================
# End to end
# =============================================================================

def check_end_to_end_against_truth(rec=_noop):
    """Full chain on the baseline geometry: every target found at the truth
    range (+- half a range bin) and Doppler (+- one bin), and the measured RD SNR
    consistent with the link budget (within straddle/window losses)."""
    from scipy.signal import get_window
    from pr450.config import load_scenario
    sc = load_scenario("scenario_baseline.yaml")
    wf = generate_lte(LteConfig(n_ports=2), "full", 0.2, seed=0)
    sim = simulate(wf, sc, seed=0)
    ref, _, _ = demod_remod(sim.ref, wf)
    canc = EcaCanceller(ref, int(np.ceil(sim.levels_db["clutter_max_delay_samples"])) + 24, n_lead=16)
    caf_kw = dict(doppler_max_hz=250, window="blackmanharris", range_window="blackmanharris", occupied_bw_hz=BW)
    res = clean(sim.surv, ref, wf.fs, 512, caf_kw, dict(pfa=1e-6, min_delay_samples=2, exclude_zero_doppler_hz=3.0),
                threshold_db=18.0, canceller=lambda v: canc(v).output)
    df = 1 / (len(ref) / wf.fs)
    nb = (len(ref) - 512) // 512
    loss = _window_loss(get_window("blackmanharris", nb, fftbins=False)) *         _window_loss(get_window("blackmanharris", 300, fftbins=False))
    p_s = np.mean(np.abs(ref) ** 2)
    # Every echo is random-modulated, so it also raises the floor of all other
    # cells (target self-noise): the floor is thermal + total echo power.
    echo_power = sum(10 ** (t.snr_db / 10) * p_s for t in sim.truth)
    floor_rise_db = _db(1.0 + echo_power)

    # RD SNR of each target in the map before CLEAN, against a target-free noise floor
    rd = caf_batched(canc(sim.surv).output, ref, wf.fs, 512, **caf_kw)
    p = rd.power()
    keep = np.ones(p.shape, dtype=bool)
    for tr in sim.truth:
        i0, j0 = rd.cell(tr.delay_s, tr.doppler_hz)
        keep[max(0, i0 - 30):i0 + 31, :] = False
        keep[:, max(0, j0 - 15):j0 + 16] = False
    keep[:, np.abs(rd.doppler_hz) < 10] = False
    floor = np.median(p[keep]) / np.log(2)              # median of exponential = ln2 * mean

    for tr in sim.truth:
        comp = min(res.components, key=lambda c: abs(c.delay_samples - tr.delay_samples) + abs(c.doppler_hz - tr.doppler_hz),
                   default=None)
        assert comp is not None, f"{tr.name} not found"
        dr = abs(comp.delay_samples - tr.delay_samples) / wf.fs * C
        dfd = abs(comp.doppler_hz - tr.doppler_hz)
        rec(f"{tr.name} range error [m]", dr, "< 19.5")
        rec(f"{tr.name} Doppler error [Hz]", dfd, f"< {2 * df:.1f}")
        assert dr < 19.5, f"{tr.name}: range error {dr:.1f} m"
        assert dfd < 2 * df, f"{tr.name}: Doppler error {dfd:.2f} Hz"
        assert comp.residual_drop_db > 15, f"{tr.name}: CLEAN only removed {comp.residual_drop_db:.1f} dB"

        i0, j0 = rd.cell(tr.delay_s, tr.doppler_hz)
        measured = _db(p[i0 - 1:i0 + 2, j0 - 1:j0 + 2].max() / floor)
        predicted = tr.snr_db + _db(p_s * nb * 512 * loss) - floor_rise_db
        # measured may fall below theory by range/Doppler straddle (<~3 dB) and the
        # residual-clutter/ADC floor (~1 dB); it must never exceed theory
        rec(f"{tr.name} RD SNR measured / link budget [dB]", round(measured, 2), round(predicted, 2))
        assert predicted - 4.5 < measured < predicted + 1.0, f"{tr.name}: {measured:.1f} dB vs budget {predicted:.1f} dB"
