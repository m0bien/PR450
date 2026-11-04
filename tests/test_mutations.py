"""
Mutation tests: do the self-checks actually catch bugs?

Each test injects one deliberate, realistic bug into the code (via
monkeypatch, undone afterwards) and asserts that the corresponding analytic
check FAILS. If a mutation survives, the self-test has a blind spot.
"""
from importlib import import_module

import numpy as np
import pytest

import analytic_checks as ac

# import_module, because pr450.core re-exports a function called `eca` that
# shadows the pr450.core.eca submodule attribute
eca_mod = import_module("pr450.core.eca")
caf_mod = import_module("pr450.core.caf")
cfar_mod = import_module("pr450.core.cfar")
geo_mod = import_module("pr450.scenario.geometry")
base_mod = import_module("pr450.waveform.base")

pytestmark = pytest.mark.stage("mutation")


def _assert_caught(check):
    with pytest.raises(AssertionError):
        check()


def test_mirrored_spectrum_is_caught(monkeypatch):
    """Subcarriers mapped to -f instead of +f: survives the modulate/demodulate round trip."""
    orig = base_mod.OfdmNumerology.subcarrier_offsets
    monkeypatch.setattr(base_mod.OfdmNumerology, "subcarrier_offsets", lambda self: -orig(self))
    _assert_caught(ac.check_subcarrier_frequency_mapping)


def test_cyclic_prefix_off_by_one_is_caught(monkeypatch):
    monkeypatch.setattr(base_mod.OfdmNumerology, "cp_short", property(lambda self: 144 * self.nfft // 2048 + 1))
    _assert_caught(ac.check_frame_timing_per_spec)


def test_doppler_sign_error_is_caught(monkeypatch):
    orig = geo_mod.bistatic_range_doppler

    def flipped(*args):
        rb, fd, r_tx, r_rx = orig(*args)
        return rb, -fd, r_tx, r_rx

    monkeypatch.setattr(ac, "bistatic_range_doppler", flipped)
    _assert_caught(ac.check_bistatic_doppler_physics)


def test_radar_equation_error_is_caught(monkeypatch):
    """E.g. 20 log10 instead of 30 log10 on the (4 pi) term."""
    def wrong(eirp_dbm, gain_dbi, rcs_dbsm, r_tx, r_rx, wavelength_m):
        return (eirp_dbm + gain_dbi + rcs_dbsm + 20 * np.log10(wavelength_m) - 20 * np.log10(4 * np.pi)
                - 20 * np.log10(r_tx) - 20 * np.log10(r_rx))
    monkeypatch.setattr(ac, "bistatic_echo_dbm", wrong)
    _assert_caught(ac.check_radar_equation_monostatic_limit)


def test_eca_lag_misalignment_is_caught(monkeypatch):
    """Interference estimate shifted by one sample (a classic indexing bug)."""
    orig = eca_mod.fftconvolve

    def shifted(a, b, *args, **kw):
        return np.roll(orig(a, b, *args, **kw), 1)
    monkeypatch.setattr(eca_mod, "fftconvolve", shifted)
    _assert_caught(ac.check_eca_orthogonality_and_idempotence)


def test_cfar_threshold_error_is_caught(monkeypatch):
    """Threshold multiplier without the '-1' term: alpha = N * Pfa^(-1/N)."""
    orig = cfar_mod.ca_cfar

    def wrong(power, guard=(2, 3), train=(4, 8), pfa=1e-6):
        gd, gf = guard
        td, tf = train
        n = (2 * (gd + td) + 1) * (2 * (gf + tf) + 1) - (2 * gd + 1) * (2 * gf + 1)
        # equivalent nominal Pfa that reproduces alpha = N * pfa^(-1/N)
        alpha = n * pfa ** (-1 / n)
        return orig(power, guard, train, (1 + alpha / n) ** (-n))
    monkeypatch.setattr(ac, "ca_cfar", wrong)
    _assert_caught(ac.check_cfar_false_alarm_rate)


def test_caf_batch_misalignment_is_caught(monkeypatch):
    """Reference batches paired with the wrong surveillance batches."""
    orig_view = caf_mod.sliding_window_view

    def misaligned(x, n):
        return orig_view(np.roll(x, 512), n)
    monkeypatch.setattr(caf_mod, "sliding_window_view", misaligned)
    _assert_caught(ac.check_caf_processing_gain)


def test_thermal_noise_error_is_caught(monkeypatch):
    """Noise power off by 3 dB (e.g. real instead of complex noise bandwidth)."""
    monkeypatch.setattr(ac, "noise_power_dbm", lambda fs, nf, t=290.0: geo_mod.noise_power_dbm(fs, 0.0, t) + 3.0)
    _assert_caught(ac.check_thermal_noise_and_friis)
