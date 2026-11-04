"""Pytest wrappers for tests/analytic_checks.py, labelled by pipeline stage."""
import pytest

import analytic_checks as ac


def _run(check, record_property):
    def rec(name, measured, expected):
        record_property(name, f"measured {measured} | expected {expected}")
    check(rec)


CASES = [
    ("waveform", ac.check_subcarrier_frequency_mapping),
    ("waveform", ac.check_frame_timing_per_spec),
    ("waveform", ac.check_load_power_consistency),
    ("receiver", ac.check_thermal_noise_and_friis),
    ("receiver", ac.check_radar_equation_monostatic_limit),
    ("receiver", ac.check_bistatic_doppler_physics),
    ("receiver", ac.check_simulated_echo_matches_truth),
    ("receiver", ac.check_adc_sqnr),
    ("reference", ac.check_channel_estimate_matches_true_channel),
    ("reference", ac.check_pilot_only_energy),
    ("reference", ac.check_remod_exact_under_snr),
    ("eca", ac.check_eca_orthogonality_and_idempotence),
    ("eca", ac.check_eca_fast_path_equals_generic),
    ("eca", ac.check_eca_preserves_moving_target),
    ("caf", ac.check_caf_processing_gain),
    ("caf", ac.check_caf_implementations_agree),
    ("cfar", ac.check_cfar_false_alarm_rate),
    ("cfar", ac.check_cfar_detection_probability),
    ("clean", ac.check_clean_noop_without_targets),
    ("clean", ac.check_clean_does_not_disturb_other_targets),
]


@pytest.mark.parametrize("stage,check", [pytest.param(s, c, marks=pytest.mark.stage(s), id=c.__name__)
                                         for s, c in CASES])
def test_check(stage, check, record_property):
    _run(check, record_property)


@pytest.mark.stage("end_to_end")
@pytest.mark.slow
def test_end_to_end_against_truth(record_property):
    _run(ac.check_end_to_end_against_truth, record_property)
