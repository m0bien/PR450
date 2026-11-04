"""
Cross-check the bit-exact LTE parts of pr450 (CRS, PSS, SSS) against srsRAN 4G.

Reference vectors in tests/data/srsran/ are produced by srsRAN's own PHY
library (tools/srsran/build_and_dump.sh, run under WSL/Linux): one 10 ms
frame per cell ID and CRS port count, 5 MHz, normal CP, FDD. Every non-zero
RE is compared, positions and values (float32 precision).
"""
from pathlib import Path

import numpy as np
import pytest

from pr450.waveform import RE_PILOT, RE_SYNC, LteConfig, generate_lte

DATA = Path(__file__).parent / "data" / "srsran"
RECORD = np.dtype([("kind", "<i4"), ("symbol", "<i4"), ("k", "<i4"), ("re", "<f4"), ("im", "<f4")])
CASES = sorted(
    (int(p.stem.split("_")[1][1:]), int(p.stem.split("_")[2][1:])) for p in DATA.glob("lte_c*_p*.bin")
)

if not CASES:
    pytest.skip("no srsRAN vectors; run tools/srsran/build_and_dump.sh under WSL", allow_module_level=True)


def _reference_grid(cell_id: int, ports: int, kind: int) -> np.ndarray:
    rec = np.fromfile(DATA / f"lte_c{cell_id}_p{ports}.bin", dtype=RECORD)
    rec = rec[rec["kind"] == kind]
    grid = np.zeros((140, 300), dtype=complex)
    grid[rec["symbol"], rec["k"]] = rec["re"] + 1j * rec["im"]
    return grid


@pytest.fixture(scope="module", params=CASES, ids=[f"cell{c}-ports{p}" for c, p in CASES])
def case(request):
    cell_id, ports = request.param
    wf = generate_lte(LteConfig(cell_id=cell_id, n_ports=ports), "idle", 0.01)
    return cell_id, ports, wf


def test_crs_positions_and_values(case):
    cell_id, ports, wf = case
    ref = _reference_grid(cell_id, ports, 1)
    mine = np.where(wf.re_type == RE_PILOT, wf.grid, 0)
    np.testing.assert_array_equal(mine != 0, ref != 0, err_msg="CRS positions differ")
    np.testing.assert_allclose(mine, ref, atol=1e-6, err_msg="CRS values differ")


def test_pss_sss_positions_and_values(case):
    """SSS is bit-exact. srsRAN builds the PSS Zadoff-Chu phase in float32
    (arg = pi*u*n(n+1)/63 reaches ~6600 rad, float32 resolution ~5e-4 rad),
    so PSS is compared by phase with that tolerance; pr450's PSS is checked
    against the exact integer-reduced phase in test_waveform.py."""
    cell_id, ports, wf = case
    ref = _reference_grid(cell_id, ports, 2)
    mine = np.where(wf.re_type == RE_SYNC, wf.grid, 0)
    np.testing.assert_array_equal(mine != 0, ref != 0, err_msg="PSS/SSS positions differ")

    pss_symbols = [6, 76]  # last symbol of slot 0 in subframes 0 and 5
    sss_symbols = [5, 75]
    np.testing.assert_allclose(mine[sss_symbols], ref[sss_symbols], atol=1e-6, err_msg="SSS values differ")
    on = ref[pss_symbols] != 0
    np.testing.assert_allclose(np.abs(mine[pss_symbols][on]), 1, atol=1e-6)
    phase_err = np.angle(mine[pss_symbols][on] / ref[pss_symbols][on])
    assert np.max(np.abs(phase_err)) < 5e-4, f"PSS phase error {np.max(np.abs(phase_err)):.2e} rad"
