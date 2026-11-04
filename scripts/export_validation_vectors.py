"""
Export the bit-exact parts of the LTE / NR generators to a .mat file so
matlab/validate_pr450.m can compare them against LTE Toolbox and 5G Toolbox.

    python scripts/export_validation_vectors.py
    matlab -batch "run('matlab/validate_pr450.m')"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pr450.waveform import RE_PILOT, RE_SYNC, LteConfig, generate_lte, gold_sequence  # noqa: E402
from pr450.waveform.nr import nr_ssb_block  # noqa: E402

CELL_IDS = (0, 1, 101, 503)
GOLD_C_INITS = (0, 1, 12345, 2**30 + 7, 2**31 - 1)
OUT = Path(__file__).resolve().parents[1] / "matlab" / "pr450_vectors.mat"


def main():
    out = {"cell_ids": np.array(CELL_IDS, dtype=float)}
    rng = np.random.default_rng(0)
    for cell in CELL_IDS:
        for n_ports in (1, 2):
            wf = generate_lte(LteConfig(cell_id=cell, n_ports=n_ports, sfn0=0), "idle", 0.01, seed=0)
            crs = np.where(wf.re_type == RE_PILOT, wf.grid, 0)
            sync = np.where(wf.re_type == RE_SYNC, wf.grid, 0)
            # MATLAB grids are K x L (subcarrier x symbol), one frame = 140 symbols
            out[f"lte_crs_c{cell}_p{n_ports}"] = crs.T
            out[f"lte_sync_c{cell}_p{n_ports}"] = sync.T
        for i_ssb in range(4):
            vals, types, _ = nr_ssb_block(cell, i_ssb, rng)
            known = np.isin(types, (RE_PILOT, RE_SYNC))
            out[f"nr_ssb_c{cell}_i{i_ssb}"] = np.where(known, vals, 0).T  # 240 x 4
    out["gold_c_inits"] = np.array(GOLD_C_INITS, dtype=float)
    out["gold_bits"] = np.stack([gold_sequence(c, 600) for c in GOLD_C_INITS]).astype(float)
    savemat(OUT, out)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
