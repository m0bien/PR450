"""
Plan Phase 2b: self-ambiguity of LTE B72 vs NR n72 at several load levels.

For each standard x load preset this computes the self-ambiguity function,
its PSLR / ISLR / sidelobe floor, the strongest deterministic artefacts,
and the load statistics that drive radar performance. Outputs go to
results/ambiguity/: one figure per standard plus metrics.md / metrics.csv.

    python scripts/run_ambiguity_study.py --duration 0.2 --loads idle,low,medium,full
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pr450.config import load_waveform_config  # noqa: E402
from pr450.eval import af_metrics, plot_cuts, plot_rd, self_ambiguity, waveform_stats  # noqa: E402
from pr450.waveform import generate  # noqa: E402

CONFIGS = {"LTE": "lte_b72_5mhz.yaml", "NR": "nr_n72_5mhz.yaml"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=0.2, help="record length [s]")
    ap.add_argument("--loads", default="idle,low,medium,full")
    ap.add_argument("--max-delay-us", type=float, default=100.0)
    ap.add_argument("--doppler-max", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "ambiguity"))
    args = ap.parse_args()

    loads = [s.strip() for s in args.loads.split(",") if s.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for std, cfg_file in CONFIGS.items():
        cfg = load_waveform_config(cfg_file)
        fig, axes = plt.subplots(len(loads), 3, figsize=(16, 3.4 * len(loads)), squeeze=False,
                                 layout="constrained")
        for r, load in enumerate(loads):
            t0 = time.time()
            wf = generate(cfg, load, args.duration, args.seed)
            rd = self_ambiguity(wf, args.max_delay_us * 1e-6, args.doppler_max)
            m = af_metrics(rd)
            st = waveform_stats(wf)
            rows.append({"standard": std, "load": load, **st,
                         **{k: m[k] for k in ("pslr_db", "islr_db", "floor_db")},
                         "top_artefact": _fmt_peak(m["peaks"][0]) if m["peaks"] else ""})
            print(f"{std:3s} {load:7s} occ={st['occupied_fraction']:.3f} P={st['mean_power_db']:6.1f} dB "
                  f"PSLR={m['pslr_db']:6.1f} ISLR={m['islr_db']:6.1f} floor={m['floor_db']:6.1f}  "
                  f"top: {rows[-1]['top_artefact']}  ({time.time() - t0:.1f}s)")

            im = plot_rd(axes[r, 0], rd, f"{std} {load}: |AF|^2 (PSLR {m['pslr_db']:.1f} dB)", dyn_range_db=60)
            plot_cuts(axes[r, 1], axes[r, 2], rd, load)
            for p in m["peaks"][:4]:
                axes[r, 0].plot(p["doppler_hz"], p["bistatic_km"], "x", color="red", ms=6)
            axes[r, 1].set_ylim(-70, 2)
            axes[r, 2].set_ylim(-70, 2)
        fig.colorbar(im, ax=axes[:, 0], location="left", shrink=0.5, label="dB rel. peak")
        fig.suptitle(f"{std} self-ambiguity, {args.duration * 1e3:.0f} ms record  (red x = strongest artefacts)")
        path = out_dir / f"ambiguity_{std.lower()}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {path}")

    _write_tables(rows, out_dir)


def _fmt_peak(p: dict) -> str:
    return f"{p['level_db']:.1f} dB @ {p['bistatic_km']:.2f} km / {p['doppler_hz']:+.0f} Hz"


def _write_tables(rows: list[dict], out_dir: Path):
    keys = list(rows[0].keys())
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "metrics.md", "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n")
        for r in rows:
            f.write("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values()) + " |\n")
    print(f"  -> {out_dir / 'metrics.md'}")


if __name__ == "__main__":
    main()
