"""
Stage labelling for the self-test.

Tests can carry an explicit @pytest.mark.stage(name); everything else is
labelled from its module and name by STAGE_RULES so that the self-test
report can give a verdict per pipeline stage.
"""
import re

import pytest

STAGES = {
    "conformance": "3GPP conformance (vs py3gpp, srsRAN 4G)",
    "waveform": "LTE / NR waveform generation",
    "receiver": "Receiver model: geometry, link budget, noise, ADC",
    "reference": "Reference reconstruction (channel estimate, demod-remod)",
    "eca": "Direct-path / clutter cancellation (ECA)",
    "caf": "Range-Doppler processing (CAF, windows, normalisation)",
    "cfar": "Detection (CA-CFAR)",
    "clean": "CLEAN target subtraction",
    "end_to_end": "End-to-end chain against ground truth",
    "mutation": "Self-test sensitivity (injected bugs caught)",
}

STAGE_RULES = [
    (r"test_against_(py3gpp|srsran)", r".*", "conformance"),
    (r"test_waveform", r".*", "waveform"),
    (r"test_processing", r"bistatic|simulation", "receiver"),
    (r"test_processing", r"remod|pilot_only", "reference"),
    (r"test_processing", r"eca", "eca"),
    (r"test_processing", r"caf|range_taper|normalisation", "caf"),
    (r"test_processing", r"cfar", "cfar"),
    (r"test_processing", r"clean", "clean"),
]


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="also run tests marked slow")


def pytest_configure(config):
    config.addinivalue_line("markers", "stage(name): pipeline stage the test verifies")
    config.addinivalue_line("markers", "slow: long-running test (needs --runslow)")


def stage_of(item) -> str:
    marker = item.get_closest_marker("stage")
    if marker:
        return marker.args[0]
    module = item.module.__name__.rsplit(".", 1)[-1]
    for mod_pat, name_pat, stage in STAGE_RULES:
        if re.search(mod_pat, module) and re.search(name_pat, item.name):
            return stage
    return "other"


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="slow: run with --runslow (or the full self-test)")
    for item in items:
        if not item.get_closest_marker("stage"):
            item.add_marker(pytest.mark.stage(stage_of(item)))
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)
