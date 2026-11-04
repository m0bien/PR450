"""
Pd / Pfa Monte Carlo study: detection performance of the processing chain,
per standard (LTE B72, NR n72), load (idle / medium / full) and processing
setting, and the detection range it implies.

Method
  * Each 0.5 s CPI carries 36 weak test echoes (6 bistatic ranges x 6 Dopplers,
    spaced wider than the CFAR window, random sub-bin offsets, random phase),
    on top of the baseline clutter field, direct path, noise and 12-bit ADC.
  * Six SNR levels per CPI (a Latin square over the 6x6 grid, so every level
    sees every range and Doppler), two interleaved level sets across CPIs.
  * Full chain: demod-remod reference -> ECA -> CAF -> CA-CFAR (Pfa 1e-6).
    All processing settings run on the same ECA output (paired comparison).
  * Pd(SNR) is fitted with a probit model (maximum likelihood); SNR for
    Pd = 0.5 / 0.9 with bootstrap 90 % intervals. Processing loss = required
    SNR minus that of an ideal CA-CFAR detector at full coherent gain.
  * Pfa: unassociated detections counted at nominal 1e-6 and 1e-4.
  * Detection range from the link budget (free space, baseline geometry).

    python scripts/run_pd_pfa_study.py                 # full study, ~10-20 min on 10 workers
    python scripts/run_pd_pfa_study.py --cpis 2 --loads full --stds LTE   # smoke run
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pr450.config import load_scenario, load_waveform_config  # noqa: E402
from pr450.core import EcaCanceller, caf_batched, detect  # noqa: E402
from pr450.reference import demod_remod  # noqa: E402
from pr450.scenario import C, InjectedEcho, bistatic_echo_dbm, noise_power_dbm, simulate  # noqa: E402
from pr450.waveform import generate  # noqa: E402

CONFIGS = {"LTE": "lte_b72_5mhz.yaml", "NR": "nr_n72_5mhz.yaml"}
BW = 300 * 15e3
SETTINGS = {
    "S0": ("Hann Doppler, no range taper, alpha 0", dict(window="hann")),
    "S1": ("Blackman-Harris both, alpha 0.5 (current; normalisation floor 0.1)", dict(window="blackmanharris", range_window="blackmanharris",
                                                           normalize=0.5)),
    "S2": ("Blackman-Harris both, alpha 0", dict(window="blackmanharris", range_window="blackmanharris")),
    "S3": ("Taylor -40 dB both, alpha 0.25", dict(window=("taylor", 4, 40), range_window=("taylor", 4, 40),
                                                normalize=0.25)),
    "S4": ("Hann both, alpha 0.5", dict(window="hann", range_window="hann", normalize=0.5)),
    "S5": ("Taylor -40 dB both, alpha 0 (adopted)", dict(window=("taylor", 4, 40), range_window=("taylor", 4, 40))),
    "S6": ("S5 + batch reciprocal filter, reg 1", dict(window=("taylor", 4, 40), range_window=("taylor", 4, 40),
                                                      mode="reciprocal", reg=1.0)),
    "S7": ("S5 + batch reciprocal filter, reg 0.1", dict(window=("taylor", 4, 40), range_window=("taylor", 4, 40),
                                                        mode="reciprocal", reg=0.1)),
}
RANGE_BINS = (140, 200, 260, 320, 380, 440)        # 5.5 .. 17.2 km bistatic, clear of the clutter zone
DOPPLERS = (-200.0, -120.0, -40.0, 40.0, 120.0, 200.0)
LEVEL_SETS = (np.array([-6.0, -3.0, 0.0, 3.0, 6.0, 9.0]), np.array([-4.5, -1.5, 1.5, 4.5, 7.5, 10.5]))
MAP_SNR_CENTRE_DB = 13.0                           # level offsets are relative to this map SNR
TARGETS_RCS = {"small drone": -10.0, "car": 10.0, "large vehicle": 20.0}   # assumed RCS at 450 MHz [dBsm]
MAX_DELAY = 512
DET_KW = dict(min_delay_samples=2, exclude_zero_doppler_hz=3.0)


def run_cpi(job: dict) -> dict:
    t0 = time.time()
    std, load, k, seed, duration = job["std"], job["load"], job["cpi"], job["seed"], job["duration"]
    wf = generate(load_waveform_config(CONFIGS[std]), load, duration, seed)
    fs = wf.fs
    p_s = float(np.mean(np.abs(wf.iq) ** 2))
    n_proc = (len(wf.iq) - MAX_DELAY) // 512 * 512
    ideal_gain_db = 10 * np.log10(p_s * n_proc)
    levels = MAP_SNR_CENTRE_DB - ideal_gain_db + LEVEL_SETS[k % 2]

    rng = np.random.default_rng(10_000 + seed)
    perm_r, perm_c, perm_l = rng.permutation(6), rng.permutation(6), rng.permutation(6)
    injected = []
    for i in range(6):
        for j in range(6):
            lvl = levels[perm_l[(i + j) % 6]]
            rb = (RANGE_BINS[perm_r[i]] + rng.random()) * C / fs
            fd = DOPPLERS[perm_c[j]] + rng.uniform(-5, 5)
            injected.append(InjectedEcho(f"t{i}{j}", rb, fd, float(lvl)))

    sc = load_scenario("scenario_baseline.yaml")
    sc.targets = []
    sc.injected = injected
    sim = simulate(wf, sc, seed)
    ref, _, _ = demod_remod(sim.ref, wf)
    canceller = EcaCanceller(ref, int(np.ceil(sim.levels_db["clutter_max_delay_samples"])) + 24, n_lead=16)
    x = canceller(sim.surv).output

    trials, fa = [], []
    for key, (_, kw) in SETTINGS.items():
        kw = dict(kw)
        if "range_window" in kw:
            kw["occupied_bw_hz"] = BW
        rd = caf_batched(x, ref, fs, MAX_DELAY, doppler_max_hz=250, **kw)
        df = abs(rd.doppler_hz[1] - rd.doppler_hz[0])
        dr = C / fs
        for pfa_nom in (1e-6, 1e-4):
            dets = detect(rd, pfa=pfa_nom, **DET_KW)
            used = set()
            for tr in sim.truth:
                hit = None
                for d in dets:
                    if abs(d.bistatic_range_m - tr.bistatic_range_m) <= 1.5 * dr and abs(d.doppler_hz - tr.doppler_hz) <= 2.5 * df:
                        hit = d if hit is None or d.snr_db > hit.snr_db else hit
                # every detection in the target's neighbourhood belongs to the target, not to the FA count
                for idx, d in enumerate(dets):
                    if abs(d.bistatic_range_m - tr.bistatic_range_m) <= 4 * dr and abs(d.doppler_hz - tr.doppler_hz) <= 6 * df:
                        used.add(idx)
                if pfa_nom == 1e-6:
                    trials.append(dict(std=std, load=load, setting=key, cpi=k, snr_db=tr.snr_db,
                                       detected=int(hit is not None), cfar_snr_db=hit.snr_db if hit else np.nan))
            n_cells = rd.data.size - 2 * rd.data.shape[1] - int(np.sum(np.abs(rd.doppler_hz) <= 3.0)) * rd.data.shape[0]
            n_cells -= len(sim.truth) * 9 * 13      # target neighbourhoods excluded above
            fa.append(dict(std=std, load=load, setting=key, cpi=k, pfa_nominal=pfa_nom,
                           false_alarms=len(dets) - len(used), cells=n_cells))
    return dict(trials=trials, fa=fa, p_s=p_s, n_proc=n_proc, seconds=time.time() - t0, job=job)


# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------

def probit_fit(snr: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy.optimize import minimize
    from scipy.stats import norm

    def nll(p):
        mu, log_s = p
        z = (snr - mu) / np.exp(log_s)
        return -np.sum(y * norm.logcdf(z) + (1 - y) * norm.logsf(z))

    mu0 = snr[np.argmin(np.abs(y.astype(float) - 0.5))] if 0 < y.mean() < 1 else snr.mean()
    res = minimize(nll, x0=[mu0, np.log(1.5)], method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6))
    return float(res.x[0]), float(np.exp(res.x[1]))


def required_snr(snr, y, pd, n_boot=200, seed=0):
    from scipy.stats import norm
    mu, s = probit_fit(snr, y)
    point = mu + norm.ppf(pd) * s
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(snr), len(snr))
        m, sb = probit_fit(snr[idx], y[idx])
        boots.append(m + norm.ppf(pd) * sb)
    lo, hi = np.percentile(boots, [5, 95])
    return point, float(lo), float(hi)


def ideal_map_snr(pd: float, pfa: float = 1e-6, n_train: int = 120) -> float:
    """Map SNR an ideal CA-CFAR (N training cells) needs for a non-fluctuating target."""
    from scipy.optimize import brentq
    from scipy.stats import ncx2
    alpha = n_train * (pfa ** (-1 / n_train) - 1)
    m = np.random.default_rng(1).gamma(n_train, 1.0 / n_train, 4000)   # CFAR noise estimate / true noise

    def pd_of(snr_db):
        a2 = 10 ** (snr_db / 10)
        return float(np.mean(ncx2.sf(2 * alpha * m, 2, 2 * a2))) - pd

    return brentq(pd_of, 0.0, 30.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpis", type=int, default=8, help="CPIs per standard and load")
    ap.add_argument("--duration", type=float, default=0.5)
    ap.add_argument("--stds", default="LTE,NR")
    ap.add_argument("--loads", default="idle,medium,full")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results" / "pd_pfa"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [dict(std=s, load=l, cpi=k, seed=1000 * i + k, duration=args.duration)
            for i, (s, l) in enumerate((s, l) for s in args.stds.split(",") for l in args.loads.split(","))
            for k in range(args.cpis)]
    print(f"{len(jobs)} CPIs x {len(RANGE_BINS) * len(DOPPLERS)} targets on {args.workers} workers", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_cpi, j) for j in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            r = f.result()
            results.append(r)
            j = r["job"]
            print(f"  [{n:3d}/{len(jobs)}] {j['std']} {j['load']:6s} CPI {j['cpi']}  {r['seconds']:.0f} s", flush=True)
    print(f"simulation done in {time.time() - t0:.0f} s", flush=True)

    trials = [t for r in results for t in r["trials"]]
    fas = [f for r in results for f in r["fa"]]
    for name, rows in (("trials.csv", trials), ("false_alarms.csv", fas)):
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    gain = {(r["job"]["std"], r["job"]["load"]): 10 * np.log10(r["p_s"] * r["n_proc"]) for r in results}
    ideal90 = ideal_map_snr(0.9)
    ideal50 = ideal_map_snr(0.5)
    sc = load_scenario("scenario_baseline.yaml")
    fs = 7.68e6
    lam = C / 463.5e6
    n0 = noise_power_dbm(fs, sc.noise_figure_db, sc.temperature_k)
    baseline = sc.baseline_m

    summary = []
    for (std, load), g in sorted(gain.items(), key=lambda kv: (kv[0][0], ["idle", "medium", "full"].index(kv[0][1]))):
        for key in SETTINGS:
            sel = [t for t in trials if t["std"] == std and t["load"] == load and t["setting"] == key]
            snr = np.array([t["snr_db"] for t in sel])
            y = np.array([t["detected"] for t in sel])
            s90, lo90, hi90 = required_snr(snr, y, 0.9)
            s50 = probit_fit(snr, y)[0]
            f6 = [f for f in fas if f["std"] == std and f["load"] == load and f["setting"] == key and f["pfa_nominal"] == 1e-6]
            f4 = [f for f in fas if f["std"] == std and f["load"] == load and f["setting"] == key and f["pfa_nominal"] == 1e-4]
            cells = sum(f["cells"] for f in f6)
            ranges = {}
            for tname, rcs in TARGETS_RCS.items():
                # SNR = EIRP + G + sigma + 20log(lambda) - 30log(4pi) - 20log(Rt Rr) - N0
                k_db = bistatic_echo_dbm(sc.eirp_dbm, sc.surv_gain_dbi, rcs, 1.0, 1.0, lam) - n0 - s90
                rt_rr = 10 ** (k_db / 20)
                r_behind = (-baseline + np.sqrt(baseline**2 + 4 * rt_rr)) / 2   # target on the baseline, behind Rx
                ranges[tname] = (np.sqrt(rt_rr) / 1e3, r_behind / 1e3)
            summary.append(dict(
                std=std, load=load, setting=key, trials=len(sel),
                snr90=s90, snr90_lo=lo90, snr90_hi=hi90, snr50=s50,
                loss90=s90 + g - ideal90,
                pfa6=sum(f["false_alarms"] for f in f6) / cells, fa6=sum(f["false_alarms"] for f in f6),
                pfa4=sum(f["false_alarms"] for f in f4) / sum(f["cells"] for f in f4), cells=cells,
                ranges=ranges))

    _write_report(out, summary, trials, gain, ideal50, ideal90, args)
    _plot(out, summary, trials, gain, ideal90)
    print((out / "summary.md").read_text(encoding="utf-8"))


def _write_report(out, summary, trials, gain, ideal50, ideal90, args):
    lines = ["# Pd / Pfa Monte Carlo study", "",
             f"{len(trials) // len(SETTINGS)} target trials per setting, {args.duration} s CPI, CA-CFAR Pfa 1e-6 "
             f"(N = 120), baseline clutter + 12-bit ADC. Ideal CA-CFAR needs a map SNR of {ideal50:.1f} dB "
             f"(Pd 0.5) / {ideal90:.1f} dB (Pd 0.9) for a non-fluctuating target.", "",
             "SNR = per-sample echo SNR over receiver noise for a fully loaded carrier (link-budget quantity).", "",
             "| Std | Load | Setting | SNR for Pd 0.9 [dB] (90 % CI) | Processing loss [dB] | FA at 1e-6 (measured Pfa) | Pfa at nominal 1e-4 |",
             "|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['std']} | {s['load']} | {s['setting']} | {s['snr90']:.1f} ({s['snr90_lo']:.1f} to {s['snr90_hi']:.1f}) | "
                     f"{s['loss90']:.1f} | {s['fa6']} ({s['pfa6']:.1e}) | {s['pfa4']:.1e} |")
    lines += ["", "Settings:"] + [f"- {k}: {v[0]}" for k, v in SETTINGS.items()]
    lines += ["", "## Detection range, Pd 0.9 (free space, baseline geometry: EIRP 61 dBm, 10 dBi, NF 5 dB, 3 km baseline)", "",
              "Geometric-mean range sqrt(Rt Rr) [km] / range behind the receiver on the baseline axis [km]. "
              "RCS values are assumptions.", "",
              "These are free-space link-budget limits, not coverage. In practice they are capped by: the processed "
              "window (20 km bistatic range, 512 delay bins, configurable); radio line of sight (4/3-Earth: about "
              "28 km from a 30 m Tx and 19 km from a 10 m Rx to a 2 m ground target, about 54 km from the Rx to a "
              "drone at 100 m); and terrain and propagation losses, which are not modelled.", "",
              "| Std | Load | Setting | " + " | ".join(f"{t} ({r:+.0f} dBsm)" for t, r in TARGETS_RCS.items()) + " |",
              "|---|---|---|" + "---|" * len(TARGETS_RCS)]
    for s in summary:
        lines.append(f"| {s['std']} | {s['load']} | {s['setting']} | " +
                     " | ".join(f"{a:.1f} / {b:.1f}" for a, b in s["ranges"].values()) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(out, summary, trials, gain, ideal90):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    keys = sorted(gain, key=lambda kv: (kv[0], ["idle", "medium", "full"].index(kv[1])))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False, layout="constrained")
    for ax, (std, load) in zip(axes.ravel(), keys):
        for key in SETTINGS:
            sel = [t for t in trials if t["std"] == std and t["load"] == load and t["setting"] == key]
            snr = np.array([t["snr_db"] for t in sel])
            y = np.array([t["detected"] for t in sel])
            mu, s = probit_fit(snr, y)
            xs = np.linspace(snr.min() - 1, snr.max() + 1, 200)
            line, = ax.plot(xs, norm.cdf((xs - mu) / s), lw=2 if key == "S1" else 1.2, label=key)
            lv = np.unique(np.round(snr, 2))
            ax.plot(lv, [y[np.isclose(snr, v, atol=0.01)].mean() for v in lv], "o", ms=3, color=line.get_color())
        ax.axvline(ideal90 - gain[(std, load)], color="k", ls="--", lw=0.8)
        ax.axhline(0.9, color="grey", ls=":", lw=0.8)
        ax.set_title(f"{std} {load}  (dashed: ideal CA-CFAR, Pd 0.9)")
        ax.set_xlabel("per-sample echo SNR, full-load equivalent [dB]")
        ax.set_ylabel("Pd")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(title="setting")
    fig.suptitle("Pd vs SNR at Pfa 1e-6, 0.5 s CPI: probit fits (lines) and measured fractions (dots)")
    fig.savefig(out / "pd_curves.png", dpi=110)


if __name__ == "__main__":
    main()
