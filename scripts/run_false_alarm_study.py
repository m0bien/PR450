"""
False-alarm mitigation study: LTE B72 vs NR n72, idle / medium / full load.

Stages (each on top of the previous one, all after ECA with the
demod-remod reference):
  baseline   Hann Doppler window, no range taper            (the original chain)
  sidelobe   Blackman-Harris Doppler + range taper, energy normalisation alpha=0.5
  clean      + CLEAN (strongest-first target subtraction, ECA re-run on the
             target-free surveillance each iteration)

    python scripts/run_false_alarm_study.py --loads idle,medium,full --duration 0.5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pr450.config import load_scenario, load_waveform_config  # noqa: E402
from pr450.core import EcaCanceller, caf_batched, clean, detect  # noqa: E402
from pr450.eval import plot_rd  # noqa: E402
from pr450.reference import demod_remod  # noqa: E402
from pr450.scenario import C, simulate  # noqa: E402
from pr450.waveform import generate  # noqa: E402

CONFIGS = {"LTE": "lte_b72_5mhz.yaml", "NR": "nr_n72_5mhz.yaml"}
STAGES = ["baseline", "sidelobe", "clean"]


PROFILES = {
    # name: (Doppler window, range taper, energy-normalisation alpha)
    "bh": ("blackmanharris", "blackmanharris", 0.5),        # first version
    "taylor": (("taylor", 4, 40), ("taylor", 4, 40), 0.0),  # Pd/Pfa study recommendation (S5)
}


def caf_settings(wf, stage: str, doppler_max: float, profile: str = "bh") -> dict:
    if stage == "baseline":
        return dict(doppler_max_hz=doppler_max, window="hann")
    win, taper, alpha = PROFILES[profile]
    return dict(doppler_max_hz=doppler_max, window=win, range_window=taper,
                occupied_bw_hz=wf.num.n_sc * wf.num.scs_hz, normalize=alpha)


def associate(truth, dets, df, range_gate_m=100.0):
    hits = []
    for t in truth:
        hits.append(next((d for d in dets if abs(d.bistatic_range_m - t.bistatic_range_m) <= range_gate_m
                          and abs(d.doppler_hz - t.doppler_hz) <= 2.5 * df), None))
    return hits, len(dets) - sum(h is not None for h in hits)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loads", default="idle,medium,full")
    ap.add_argument("--duration", type=float, default=0.5)
    ap.add_argument("--scenario", default="scenario_baseline.yaml")
    ap.add_argument("--max-range-km", type=float, default=20.0)
    ap.add_argument("--doppler-max", type=float, default=250.0)
    ap.add_argument("--pfa", type=float, default=1e-6)
    ap.add_argument("--clean-threshold", type=float, default=18.0, help="CLEAN stops below this CFAR SNR [dB]")
    ap.add_argument("--plot-load", default="idle")
    ap.add_argument("--profile", choices=list(PROFILES), default="taylor", help="sidelobe-control profile")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "false_alarms"))
    args = ap.parse_args()

    loads = [s.strip() for s in args.loads.split(",") if s.strip()]
    sc = load_scenario(args.scenario)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    det_kw = dict(pfa=args.pfa, min_delay_samples=2, exclude_zero_doppler_hz=3.0)
    rows = []
    fig = axes = None

    for load in loads:
        if load == args.plot_load:
            fig, axes = plt.subplots(len(CONFIGS), len(STAGES), figsize=(5.2 * len(STAGES), 4.2 * len(CONFIGS)),
                                     squeeze=False, layout="constrained")
        for r, (std, cfg_file) in enumerate(CONFIGS.items()):
            t0 = time.time()
            wf = generate(load_waveform_config(cfg_file), load, args.duration, args.seed)
            sim = simulate(wf, sc, args.seed)
            ref, _, _ = demod_remod(sim.ref, wf)
            lv = sim.levels_db
            n_taps = int(np.ceil(lv["clutter_max_delay_samples"])) + 24
            canceller = EcaCanceller(ref, n_taps, n_lead=16)
            x = canceller(sim.surv).output
            max_delay = int(np.ceil(args.max_range_km * 1e3 / C * wf.fs))

            for c, stage in enumerate(STAGES):
                kw = caf_settings(wf, stage, args.doppler_max, args.profile)
                if stage == "clean":
                    res = clean(sim.surv, ref, wf.fs, max_delay, kw, det_kw, threshold_db=args.clean_threshold,
                                canceller=lambda v: canceller(v).output)
                    rd, dets, n_clean = res.rd, res.detections, len(res.components)
                else:
                    rd = caf_batched(x, ref, wf.fs, max_delay, **kw)
                    dets, n_clean = detect(rd, **det_kw), 0
                df = abs(rd.doppler_hz[1] - rd.doppler_hz[0])
                hits, fa = associate(sim.truth, dets, df)
                rows.append(dict(std=std, load=load, stage=stage, hits=sum(h is not None for h in hits),
                                 n_targets=len(sim.truth), false_alarms=fa, cleaned=n_clean,
                                 snr="/".join("--" if h is None else f"{h.snr_db:.0f}" for h in hits)))
                if axes is not None and load == args.plot_load:
                    title = f"{std} {load} - {stage}: {fa} false alarms"
                    if stage == "clean":
                        title += f" ({n_clean} cleaned)"
                    plot_rd(axes[r, c], rd, title, dyn_range_db=45, floor_normalise=True, truth=sim.truth,
                            detections=dets)
            print(f"{std} {load}: {time.time() - t0:.0f} s")
        if fig is not None and load == args.plot_load:
            fig.suptitle(f"False-alarm mitigation, load={load}: red o = truth, + = detections "
                         f"(CLEAN map shows the residual after target subtraction)")
            path = out_dir / f"rd_stages_{load}.png"
            fig.savefig(path, dpi=115)
            plt.close(fig)
            fig = axes = None
            print(f"-> {path}")

    lines = ["| std | load | stage | detected | false alarms | CLEAN comps | target SNR over CFAR noise [dB] |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['std']} | {r['load']} | {r['stage']} | {r['hits']}/{r['n_targets']} | "
                     f"{r['false_alarms']} | {r['cleaned']} | {r['snr']} |")
    table = "\n".join(lines)
    (out_dir / "summary.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()
