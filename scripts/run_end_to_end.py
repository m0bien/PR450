"""
End-to-end passive radar chain on simulated LTE B72 and NR n72 illuminators.

    waveform -> 2-channel receiver sim -> reference reconstruction -> ECA
             -> CAF (several reference variants) -> CA-CFAR -> truth association

ECA always uses the demod-remod reference (a pilot-only reference cannot
cancel the data-carrying direct path). The CAF is then run with:
  raw          batched CAF against the received reference channel
  demod-remod  batched CAF against the reconstructed reference
  pilot-only   batched CAF against the known-RE-only reference
  reciprocal   OFDM-domain reciprocal filter on the remodulated grid

    python scripts/run_end_to_end.py --load medium --duration 0.5
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
from pr450.core import caf_batched, caf_ofdm, detect, eca  # noqa: E402
from pr450.eval import plot_rd  # noqa: E402
from pr450.reference import demod_remod, pilot_only  # noqa: E402
from pr450.scenario import simulate  # noqa: E402
from pr450.waveform import generate  # noqa: E402

CONFIGS = {"LTE": "lte_b72_5mhz.yaml", "NR": "nr_n72_5mhz.yaml"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load", default="medium")
    ap.add_argument("--duration", type=float, default=0.5, help="CPI [s]")
    ap.add_argument("--scenario", default="scenario_baseline.yaml")
    ap.add_argument("--n-taps", type=int, default=0, help="ECA taps (0 = cover the clutter extent)")
    ap.add_argument("--eca-window", type=int, default=-1, help="-1 = classic ECA, 0 = ECA-B, k>0 = ECA-S")
    ap.add_argument("--eca-batch", type=int, default=1 << 15)
    ap.add_argument("--eca-lead", type=int, default=16, help="ECA pre-cursor (negative-lag) taps")
    ap.add_argument("--max-range-km", type=float, default=20.0)
    ap.add_argument("--doppler-max", type=float, default=250.0)
    ap.add_argument("--pfa", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "end_to_end"))
    args = ap.parse_args()

    sc = load_scenario(args.scenario)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = ["raw", "demod-remod", "pilot-only", "reciprocal"]
    fig, axes = plt.subplots(len(CONFIGS), len(variants), figsize=(5 * len(variants), 4.2 * len(CONFIGS)),
                             squeeze=False, layout="constrained")
    report = []
    summary = []

    for r, (std, cfg_file) in enumerate(CONFIGS.items()):
        t0 = time.time()
        wf = generate(load_waveform_config(cfg_file), args.load, args.duration, args.seed)
        sim = simulate(wf, sc, args.seed)
        lv = sim.levels_db
        print(f"\n=== {std} / load={args.load} / CPI={args.duration:.2f} s ===")
        print(f"ref DNR {lv['ref_dnr_db']:.1f} dB, surv DNR {lv['surv_dnr_db']:.1f} dB, "
              f"max CNR {lv['clutter_cnr_max_db']:.1f} dB, clutter extent {lv['clutter_max_delay_samples']:.0f} samples")

        ref_remod, grid_remod, st = demod_remod(sim.ref, wf)
        print(f"demod-remod: SER {st.symbol_error_rate:.2e}, missed REs {st.missed_re_fraction:.2e}, "
              f"false REs {st.false_re_fraction:.2e}")

        n_taps = args.n_taps or int(np.ceil(lv["clutter_max_delay_samples"])) + 24
        window = None if args.eca_window < 0 else args.eca_window
        res = eca(sim.surv, ref_remod, n_taps, batch_len=args.eca_batch, window=window, n_lead=args.eca_lead)
        residual_db = 10 * np.log10(np.mean(np.abs(res.output) ** 2))
        print(f"ECA ({args.eca_lead}+{n_taps} taps, window={window}): cancellation {res.cancellation_db:.1f} dB, "
              f"residual {residual_db:.1f} dB over thermal noise")

        max_delay = int(np.ceil(args.max_range_km * 1e3 / 299_792_458.0 * wf.fs))
        refs = {"raw": sim.ref, "demod-remod": ref_remod, "pilot-only": pilot_only(wf)}
        maps = {name: caf_batched(res.output, ref, wf.fs, max_delay, doppler_max_hz=args.doppler_max)
                for name, ref in refs.items()}
        maps["reciprocal"] = caf_ofdm(res.output, wf, grid_remod, min(max_delay, wf.num.nfft),
                                      doppler_max_hz=args.doppler_max)

        for c, name in enumerate(variants):
            rd = maps[name]
            dets = detect(rd, pfa=args.pfa, min_delay_samples=2, exclude_zero_doppler_hz=3.0)
            plot_rd(axes[r, c], rd, f"{std} {args.load} - {name} ({len(dets)} det.)", dyn_range_db=45,
                    floor_normalise=True, truth=sim.truth, detections=dets)
            hits = {tr.name: _associate(tr, dets, rd) for tr in sim.truth}
            used = {id(h) for h in hits.values() if h is not None}
            false_alarms = sum(1 for d in dets if id(d) not in used)
            summary.append((std, name, sum(h is not None for h in hits.values()), len(sim.truth), false_alarms))
            for tr in sim.truth:
                report.append((std, name, tr.name, tr.bistatic_range_m, tr.doppler_hz, tr.snr_db, hits[tr.name]))
        print(f"processing time {time.time() - t0:.1f} s")

    fig.suptitle(f"Range-Doppler after ECA, colour = dB over median floor, red o = truth, + = CFAR detections "
                 f"(Pfa {args.pfa:g})")
    path = out_dir / f"rd_{args.load}.png"
    fig.savefig(path, dpi=120)
    print(f"\n-> {path}\n")
    _print_report(report)
    print(f"\n{'std':4s} {'reference':12s} {'detected':>9s} {'false alarms':>13s}")
    for std, name, n_hit, n_tgt, n_fa in summary:
        print(f"{std:4s} {name:12s} {n_hit:>5d}/{n_tgt:<3d} {n_fa:>13d}")


def _associate(tr, dets, rd, range_gate_m=80.0, doppler_gate_hz=None):
    df = abs(rd.doppler_hz[1] - rd.doppler_hz[0])
    gate_f = doppler_gate_hz or 2.5 * df
    for d in dets:
        if abs(d.bistatic_range_m - tr.bistatic_range_m) <= range_gate_m and abs(d.doppler_hz - tr.doppler_hz) <= gate_f:
            return d
    return None


def _print_report(rows):
    print(f"{'std':4s} {'reference':12s} {'target':10s} {'Rb [km]':>8s} {'fd [Hz]':>8s} {'SNR/sample':>11s}  result")
    for std, ref, name, rb, fd, snr, hit in rows:
        res = f"DETECTED  {hit.snr_db:5.1f} dB over CFAR noise" if hit else "missed"
        print(f"{std:4s} {ref:12s} {name:10s} {rb / 1e3:8.2f} {fd:8.1f} {snr:9.1f} dB  {res}")


if __name__ == "__main__":
    main()
