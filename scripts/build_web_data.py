"""
Export the data behind the static web explorer (web/index.html).

GitHub Pages serves static files only, so the Python pipeline runs here,
once, and the page loads the results:

  web/data/index.json                 combinations, detection-range table, build info
  web/data/frames_<std>_<load>.json   tab 1: resource grid, power, spectrum, ambiguity, KPIs
  web/data/radar_<std>_<load>.json    tab 2: geometry, range-Doppler maps per stage, detections, KPIs

Heatmaps are stored as base64 uint8 (resource-element types; dB at 1/3 dB steps).

    python scripts/build_web_data.py            # all combinations, ~3-5 min on 6 workers
    python scripts/build_web_data.py --quick    # LTE + NR at medium load only
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse  # noqa: E402
import base64  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pr450.config import load_scenario, load_waveform_config  # noqa: E402
from pr450.core import EcaCanceller, caf_batched, clean, detect  # noqa: E402
from pr450.eval import af_metrics, self_ambiguity, waveform_stats  # noqa: E402
from pr450.reference import demod_remod  # noqa: E402
from pr450.scenario import C, random_clutter, simulate  # noqa: E402
from pr450.waveform import RE_CONTROL, RE_DATA, generate  # noqa: E402

OUT = ROOT / "web" / "data"
STANDARDS = {"LTE": ("lte_b72_5mhz.yaml", "LTE Band 72"), "NR": ("nr_n72_5mhz.yaml", "NR n72")}
LOADS = ["idle", "low", "medium", "high", "full"]
FRAME_MS = 20            # two radio frames: NR's SSB period is 20 ms
AF_MS = 200              # record length for the self-ambiguity
CPI_S = 0.5
MAX_DELAY = 512          # 20 km bistatic range
DOPPLER_MAX = 250.0
TAYLOR = ("taylor", 4, 40)
DET_KW = dict(pfa=1e-6, min_delay_samples=2, exclude_zero_doppler_hz=3.0)


def b64_u8(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode("ascii")


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))


def r(x, n=3):
    return float(np.round(float(x), n))


# -----------------------------------------------------------------------------
# Tab 1: OFDM frames
# -----------------------------------------------------------------------------

def frame_data(std: str, load: str) -> dict:
    cfg = load_waveform_config(STANDARDS[std][0])
    wf = generate(cfg, load, FRAME_MS / 1000, seed=1)
    num = wf.num
    st = waveform_stats(wf)
    occ = wf.occupancy()

    sym_power = np.sum(np.abs(wf.grid) ** 2, axis=1) / num.n_sc
    t_ms = (wf.sym_starts + wf.cp_lens + num.nfft / 2) / wf.fs * 1e3

    seg = num.nfft
    w = np.hanning(seg)
    nseg = len(wf.iq) // seg
    psd = np.mean(np.abs(np.fft.fft(wf.iq[:nseg * seg].reshape(nseg, seg) * w, axis=1)) ** 2, axis=0)
    psd = np.fft.fftshift(psd)
    f_mhz = np.fft.fftshift(np.fft.fftfreq(seg, 1 / wf.fs)) / 1e6

    rng = np.random.default_rng(0)
    payload = np.argwhere(np.isin(wf.re_type, (RE_DATA, RE_CONTROL)))
    pick = payload[rng.choice(len(payload), min(500, len(payload)), replace=False)] if len(payload) else payload
    const = wf.grid[pick[:, 0], pick[:, 1]] if len(pick) else np.array([])
    data_bits = sorted({int(b) for b in np.unique(wf.mod_bits[wf.re_type == RE_DATA])} - {0})

    wf_af = generate(cfg, load, AF_MS / 1000, seed=1)
    rd = self_ambiguity(wf_af, 100e-6, 600.0)
    m = af_metrics(rd)
    p = rd.power()
    i0, j0 = rd.cell(0.0, 0.0)
    rng_cut = db(p[:, j0] / p[i0, j0])
    dop_cut = db(p[i0, :] / p[i0, j0])

    return {
        "std": std, "load": load, "label": STANDARDS[std][1],
        "grid": {"n_sym": int(wf.n_symbols), "n_sc": int(num.n_sc), "re_type": b64_u8(wf.re_type),
                 "t_ms": [r(x) for x in t_ms],
                 "f_mhz": [r(x, 4) for x in num.subcarrier_offsets() * num.scs_hz / 1e6]},
        "sym_power_db": [r(x, 2) for x in db(sym_power + 1e-6)],
        "spectrum": {"f_mhz": [r(x) for x in f_mhz], "db": [r(x, 2) for x in db(psd / psd.max())]},
        "constellation": {"i": [r(x) for x in const.real], "q": [r(x) for x in const.imag], "data_bits": data_bits},
        "ambiguity": {
            "range_km": [r(x) for x in rd.bistatic_range_m / 1e3],
            "range_cut_db": [r(x, 2) for x in rng_cut],
            "doppler_hz": [r(x, 2) for x in rd.doppler_hz],
            "doppler_cut_db": [r(x, 2) for x in dop_cut],
            "pslr_db": r(m["pslr_db"], 2), "islr_db": r(m["islr_db"], 2),
            "peaks": [{k: r(v, 2) for k, v in pk.items()} for pk in m["peaks"][:5]],
        },
        "kpi": {
            "occupied_pct": r(100 * occ["occupied"], 1), "pilot_pct": r(100 * occ["pilot"], 2),
            "sync_pct": r(100 * occ["sync"], 2), "control_pct": r(100 * occ["control"], 1),
            "data_pct": r(100 * occ["data"], 1), "mean_power_db": r(st["mean_power_db"], 2),
            "papr_db": r(st["papr_db"], 2), "symbol_power_cv": r(st["symbol_power_cv"], 2),
            "known_energy_pct": r(100 * st["known_energy_fraction"], 1),
        },
    }


# -----------------------------------------------------------------------------
# Tab 2: passive radar
# -----------------------------------------------------------------------------

def ellipse(tx, rx, rb, n=160):
    tx, rx = np.asarray(tx[:2], float), np.asarray(rx[:2], float)
    L = np.linalg.norm(rx - tx)
    a = (rb + L) / 2
    b = np.sqrt(max(a * a - (L / 2) ** 2, 0.0))
    c = (tx + rx) / 2
    ang = np.arctan2(*(rx - tx)[::-1])
    t = np.linspace(0, 2 * np.pi, n)
    x, y = a * np.cos(t), b * np.sin(t)
    return (c[0] + x * np.cos(ang) - y * np.sin(ang)) / 1e3, (c[1] + x * np.sin(ang) + y * np.cos(ang)) / 1e3


def associate(truth, dets, df, range_gate_m=100.0):
    hits = [next((d for d in dets if abs(d.bistatic_range_m - t.bistatic_range_m) <= range_gate_m
                  and abs(d.doppler_hz - t.doppler_hz) <= 2.5 * df), None) for t in truth]
    return hits, len(dets) - sum(h is not None for h in hits)


def radar_data(std: str, load: str) -> dict:
    t0 = time.time()
    wf = generate(load_waveform_config(STANDARDS[std][0]), load, CPI_S, seed=0)
    sc = load_scenario("scenario_baseline.yaml")
    sim = simulate(wf, sc, seed=0)
    ref, grid_hat, rst = demod_remod(sim.ref, wf)
    wrong_res = int(np.count_nonzero(np.abs(grid_hat - wf.grid) > 1e-9))
    lv = sim.levels_db
    n_taps = int(np.ceil(lv["clutter_max_delay_samples"])) + 24
    canceller = EcaCanceller(ref, n_taps, n_lead=16)
    eca_res = canceller(sim.surv)
    x = eca_res.output
    fs = wf.fs
    bw = wf.num.n_sc * wf.num.scs_hz

    hann = dict(doppler_max_hz=DOPPLER_MAX, window="hann")
    taylor = dict(doppler_max_hz=DOPPLER_MAX, window=TAYLOR, range_window=TAYLOR, occupied_bw_hz=bw)

    # thermal-noise floor of each processing setting, so all maps share one scale (dB over noise)
    rng = np.random.default_rng(5)
    noise = (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))) / np.sqrt(2)
    floor = {k: float(np.mean(caf_batched(noise, ref, fs, MAX_DELAY, **kw).power())) for k, kw in
             (("hann", hann), ("taylor", taylor))}

    stages = []
    for key, title, sig, kw, fkey in (
        ("raw", "Raw: no cancellation", sim.surv, hann, "hann"),
        ("eca", "After ECA", x, hann, "hann"),
        ("sidelobe", "+ sidelobe control", x, taylor, "taylor"),
    ):
        rd = caf_batched(sig, ref, fs, MAX_DELAY, **kw)
        dets = detect(rd, **DET_KW)
        stages.append((key, title, rd, dets, fkey, []))
    res = clean(sim.surv, ref, fs, MAX_DELAY, taylor, DET_KW, threshold_db=18.0,
                canceller=lambda v: canceller(v).output)
    stages.append(("clean", "+ CLEAN", res.rd, res.detections, "taylor", res.components))

    rd0 = stages[0][2]
    df = abs(rd0.doppler_hz[1] - rd0.doppler_hz[0])
    out_stages = {}
    for key, title, rd, dets, fkey, comps in stages:
        p_db = db(rd.power() / floor[fkey])
        hits, fa = associate(sim.truth, dets, df)
        out_stages[key] = {
            "title": title,
            "map": b64_u8(np.clip(np.round(p_db * 3), 0, 255)),   # 1/3 dB steps, 0..85 dB over noise
            "peak_db": r(p_db.max(), 1),
            "detections": [{"r_km": r(d.bistatic_range_m / 1e3), "fd": r(d.doppler_hz, 2), "snr": r(d.snr_db, 1)}
                           for d in dets[:200]],
            "n_detections": len(dets), "false_alarms": int(fa), "hits": int(sum(h is not None for h in hits)),
            "target_snr": [None if h is None else r(h.snr_db, 1) for h in hits],
            "clean_components": len(comps),
        }

    targets = []
    for tgt, tr in zip(sc.targets, sim.truth):
        ex, ey = ellipse(sc.tx_position_m, sc.rx_position_m, tr.bistatic_range_m)
        targets.append({
            "name": tr.name, "x_km": r(tgt.position_m[0] / 1e3), "y_km": r(tgt.position_m[1] / 1e3),
            "vx": tgt.velocity_mps[0], "vy": tgt.velocity_mps[1], "rcs_dbsm": tgt.rcs_dbsm,
            "r_km": r(tr.bistatic_range_m / 1e3), "fd": r(tr.doppler_hz, 2), "snr_per_sample_db": r(tr.snr_db, 1),
            "speed_mps": r(np.hypot(*tgt.velocity_mps[:2]), 1),
            "ellipse": {"x": [r(v) for v in ex], "y": [r(v) for v in ey]},
        })
    clutter = [{"x_km": r(p[0] / 1e3), "y_km": r(p[1] / 1e3), "rcs": r(s, 1)}
               for p, s in random_clutter(sc.clutter, sc.rx_position_m)]

    return {
        "std": std, "load": load, "label": STANDARDS[std][1], "seconds": r(time.time() - t0, 1),
        "axes": {"range_km": [r(v) for v in rd0.bistatic_range_m / 1e3], "doppler_hz": [r(v, 2) for v in rd0.doppler_hz],
                 "n_range": int(rd0.data.shape[0]), "n_doppler": int(rd0.data.shape[1]), "db_step": 1 / 3},
        "geometry": {"tx": [sc.tx_position_m[0] / 1e3, sc.tx_position_m[1] / 1e3],
                     "rx": [sc.rx_position_m[0] / 1e3, sc.rx_position_m[1] / 1e3],
                     "targets": targets, "clutter": clutter},
        "kpi": {
            "ref_dnr_db": r(lv["ref_dnr_db"], 1), "surv_dnr_db": r(lv["surv_dnr_db"], 1),
            "clutter_cnr_db": r(lv["clutter_cnr_max_db"], 1), "remod_wrong_res": wrong_res,
            "eca_cancellation_db": r(eca_res.cancellation_db, 1),
            "eca_residual_db": r(db(np.mean(np.abs(x) ** 2)), 2), "eca_taps": n_taps + 16,
            "cpi_s": CPI_S, "pfa": DET_KW["pfa"], "cells": int(rd0.data.size),
        },
        "stages": out_stages,
    }


# -----------------------------------------------------------------------------

def detection_ranges() -> dict:
    """Pd 0.9 range table from the Monte Carlo study (results/pd_pfa/summary.md), adopted setting S5."""
    path = ROOT / "results" / "pd_pfa" / "summary.md"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (LTE|NR) \| (\w+) \| (S5) \| ([\d.]+) / [\d.]+ \| ([\d.]+) / [\d.]+ \| ([\d.]+) / [\d.]+ \|", line)
        if m:
            out.setdefault(m.group(1), {})[m.group(2)] = {"drone": float(m.group(4)), "car": float(m.group(5)),
                                                          "large_vehicle": float(m.group(6))}
    return out


def job(kind, std, load):
    data = frame_data(std, load) if kind == "frames" else radar_data(std, load)
    path = OUT / f"{kind}_{std.lower()}_{load}.json"
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return f"{path.name}: {path.stat().st_size / 1024:.0f} kB"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="medium load only")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    loads = ["medium"] if args.quick else LOADS
    jobs = [(k, s, l) for k in ("frames", "radar") for s in STANDARDS for l in loads]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for msg in pool.map(job, *zip(*jobs)):
            print("  " + msg, flush=True)
    index = {
        "standards": {k: v[1] for k, v in STANDARDS.items()}, "loads": loads,
        "built": dt.datetime.now().isoformat(timespec="seconds"),
        "detection_range_km": detection_ranges(),
        "notes": {"frame_ms": FRAME_MS, "cpi_s": CPI_S, "max_range_km": MAX_DELAY / 7.68e6 * C / 1e3,
                  "scenario": "baseline: Tx at origin, Rx 3 km east, EIRP 61 dBm, 40 clutter scatterers, 3 targets"},
    }
    (OUT / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
