# pr450 — 450 MHz LTE / 5G NR passive radar simulation framework

A dual-track simulation framework for an SDR passive radar that uses a private
450 MHz LTE / 5G NR network as its illuminator. The band (B72 / n72, DL 461–466 MHz)
and the radio access technology of such networks vary, so LTE and NR are modelled side
by side. The two tracks share everything except the waveform module.

**Interactive explorer:** [web/](web/), published via GitHub Pages. It has two views:
OFDM frames at idle to full load, and the passive radar chain step by step. See
[web/README.md](web/README.md).

```
configs/                      YAML: LTE B72, NR n72, baseline bistatic scenario
pr450/waveform/   base.py     OFDM numerology, Gold sequences, QAM, modulator, Waveform container
                  lte.py      LTE FDD DL: PSS/SSS, CRS (bit-exact) + PBCH/PDCCH/PDSCH (statistical)
                  nr.py       NR FDD DL: SSB (PSS/SSS/PBCH-DMRS), PDSCH/PDCCH DMRS, TRS (bit-exact) + payloads
                  load.py     load presets: idle, low, medium, high, full (TM1.1-like), full_64qam (TM3.1-like)
pr450/reference/              raw, demod-remod, pilot-only, perfect reference reconstruction
pr450/scenario/               bistatic geometry, link budget, clutter, 2-channel receiver + ADC model
pr450/core/       eca.py      ECA / ECA-B / ECA-S, pre-cursor taps, EcaCanceller (reference work cached)
                  caf.py      batched + OFDM-domain CAF; Doppler window, range taper, energy normalisation
                  clean.py    CLEAN: strongest-first target subtraction with ECA re-run
                  cfar.py     2-D CA-CFAR + detection extraction
pr450/eval/                   self-ambiguity metrics, waveform statistics, plotting
scripts/                      run_ambiguity_study.py, run_end_to_end.py, run_false_alarm_study.py, ...
matlab/validate_pr450.m       cross-check vs LTE Toolbox + 5G Toolbox (toolboxes not installed here)
tests/                        pytest suite (156 tests: 93 py3gpp + 28 srsRAN cross-checks)
tools/srsran/                 srsRAN 4G vector dumper (C) + WSL build script
```

## Self-test: is everything working?

```bash
python -m pr450.selftest            # all stages, ~2 min; exit code != 0 if anything fails
python -m pr450.selftest --quick    # skip the end-to-end run, ~35 s
python -m pr450.selftest --stage eca --stage caf -v
```

The runner gives a verdict per pipeline stage and writes `results/selftest/report.md` and `report.json`. These hold the measured values, the environment, and a fingerprint of code + configs. Each stage is checked against references that don't depend on the code under test:

| Stage | Checked against |
|---|---|
| Conformance | py3gpp (NR) and srsRAN 4G (LTE): sequences, RE positions, OFDM modulation |
| Waveform | Absolute subcarrier→frequency mapping (catches a mirrored spectrum, which survives round trips); 3GPP sample positions of the frame and PSS/SSB; Parseval power vs load |
| Receiver | kT0 = −173.98 dBm/Hz, FSPL formula, monostatic radar equation, bistatic Doppler sign and magnitude, simulated echo at truth delay/Doppler/power, ADC SQNR = 6.02b + 1.76 − backoff |
| Reference | Channel estimate vs exact response of a known channel (< −40 dB), Parseval for pilot-only, bit-exact remod |
| ECA | Least-squares orthogonality (1e-12) and idempotence, fast = generic path, no loss for moving targets |
| CAF | Output SNR = \|a\|²·P·N·window losses (±0.5 dB), axis scaling, batched = OFDM-domain |
| CFAR | Empirical Pfa vs nominal; Pd vs the closed-form Swerling-1 CA-CFAR formula |
| CLEAN | No-op on target-free data; removing a strong target restores a weak one's SNR (±1 dB) |
| End to end | Baseline scenario: every target within 19.5 m / 2 Doppler bins; RD SNR vs link budget including target self-noise |
| Mutation | 8 realistic bugs (mirrored spectrum, CP off by one, Doppler sign, radar-equation term, ECA lag shift, CFAR threshold, CAF batch misalignment, 3 dB noise error) are injected and must be caught |

A pass can't prove the absence of every logical error. It shows that each stage agrees with physics, closed-form results and independent implementations, and that the checks are sensitive enough to catch the listed classes of bug. Plain `pytest` runs the same tests; add `--runslow` to include the end-to-end check.

## Quick start

```bash
pip install -e .[test,xcheck]   # xcheck = py3gpp (AGPL, test-only)
python -m pytest
python scripts/run_ambiguity_study.py --duration 0.2 --loads idle,low,medium,full
python scripts/run_end_to_end.py --load medium --duration 0.5
python scripts/run_false_alarm_study.py --loads idle,medium,full
```

Results are written to `results/ambiguity/` and `results/end_to_end/`.

## Shared vs. standard-specific

At 5 MHz, LTE and NR (µ=0) share the subcarrier spacing (15 kHz), FFT size (512),
sample rate (7.68 Msps), occupied bandwidth (300 subcarriers ≈ 4.5 MHz, so ≈ 67 m
bistatic range resolution) and the normal-CP pattern. Only the resource-grid content
differs. LTE has an always-on CRS and leaves the DC subcarrier empty. NR has no
always-on reference signal: an idle cell sends only SSB every 20 ms, plus SIB1 and TRS.

## First results (simulated, unconfirmed network parameters)

**Self-ambiguity** (`run_ambiguity_study.py`, 200 ms):

| | idle | low | medium | full |
|---|---|---|---|---|
| LTE occupied REs / PSLR | 12.6 % / −7.0 dB | 20 % / −11.4 dB | 38 % / −16.8 dB | 95 % / −17.9 dB |
| NR occupied REs / PSLR | 2.2 % / −1.2 dB | 11 % / −13.3 dB | 30 % / −16.6 dB | 90 % / −18.2 dB |

The artefacts match the predictions:
- **LTE:** CRS-comb range peaks at 6.6 km and 13.3 km (Tu/3, 2Tu/3), and 200 Hz Doppler lines from PSS/SSS.
- **NR:** a 50 Hz Doppler comb from the 20 ms SSB period, which dominates when idle.
- **Both:** the CP replica at 20 km (Tu = 66.7 µs).

**End-to-end** (`run_end_to_end.py`, 0.5 s CPI, baseline geometry, 12-bit ADC):
- **Reference reconstruction:** demod-remod is bit-exact for every load on both standards.
- **Cancellation:** ECA removes the direct path (+51 dB DNR) and clutter (up to +48 dB CNR) to within ~1–3 dB of thermal noise.
- **Detection:** all 3 targets (car, drone, far truck) are detected for every standard, load and reference variant.
- **False alarms** (first version): about 45–200 per map at Pfa = 1e-6. They came from target sidelobes, not noise.

**False-alarm mitigation** (`run_false_alarm_study.py`, 0.5 s CPI, Pfa 1e-6, 128k cells per map):

| False alarms per map | idle | medium | full |
|---|---|---|---|
| LTE: baseline → sidelobe control → + CLEAN | 147 → 9 → **0** | 92 → 2 → **0** | 112 → 17 → **0** |
| NR: baseline → sidelobe control → + CLEAN | 85 → 85 → **1** | 56 → 21 → **1** | 116 → 23 → **1** |

All 3 targets are detected in every case.

What caused the false alarms, measured before any fix:

| Share | Cause | Fix |
|---|---|---|
| ~60–95 % | Range ridge of strong targets: the occupied spectrum (300 of 512 bins) is sharp-edged, giving slowly decaying sinc-like range sidelobes | Spectral range taper (Blackman-Harris over the occupied band) |
| ~15–35 % | Doppler ridge: bursty load amplitude-modulates every echo | Per-batch energy equalisation (mean/E)^α, α = 0.5 |
| Most of idle NR | Deterministic comb ghosts: PBCH-DMRS every 4th subcarrier gives ±5 km; SSB period gives a 50 Hz comb. LTE's CRS comb gives ±6.6 km | CLEAN. No window can touch these |

**Costs and remaining issues:**
- **SNR:** sidelobe control costs about 4–10 dB of target SNR against the CFAR noise estimate (window losses, plus noise weighting from normalisation). The trade-off needs tuning by Pd/Pfa Monte Carlo, for example lighter Taylor windows or α < 0.5.
- **NR residual false alarm:** a second-order comb ghost of the car's subtraction residual (+2 × 5.1 km, −5 × 50 Hz).
- **CLEAN:** about 40 s per CPI in NumPy, dominated by the 51-column echo fit.

**Pd/Pfa Monte Carlo** (`run_pd_pfa_study.py`). The study used 1,728 weak-target trials per setting, through the full chain with clutter and a 12-bit ADC, at Pfa 1e-6 and a 0.5 s CPI. Detailed results are in `results/pd_pfa/`. "Loss" below is the SNR needed for Pd 0.9 relative to an ideal CA-CFAR.

| Setting | LTE loss idle / med / full [dB] | NR loss idle / med / full [dB] |
|---|---|---|
| S0 Hann Doppler, no taper | 3.5 / 3.7 / 3.1 | 5.7 / 4.1 / 3.4 |
| S1 Blackman-Harris both, α 0.5 (first version) | 8.2 / 7.7 / 7.3 | 9.5 / 7.1 / 6.6 |
| **S5 Taylor −40 dB both, α 0, matched filter (adopted)** | 4.2 / 4.8 / 3.9 | 6.1 / 4.2 / 3.7 |
| S6 S5 + batch reciprocal filter, reg 1 | 4.5 / 5.0 / 4.0 | 7.4 / 4.8 / 4.0 |
| S7 S5 + batch reciprocal filter, reg 0.1 | 7.8 / 8.0 / 5.7 | 10.4 / 7.9 / 5.3 |

- **S5 is the default** (`run_false_alarm_study.py --profile taylor`). With CLEAN and strong targets present it gives 0–2 false alarms per map.
- **Batch reciprocal filter** (`caf_batched(mode="reciprocal")`, a data-aided per-batch CSI estimate): kept as an option, not adopted.
  - With strong targets and no CLEAN, it lowers target self-noise by 0.6–1.6 dB and gains only 0.1–0.7 dB of target SNR.
  - On weak targets it costs 0.1–1.3 dB (reg 1) or 1.3–4.3 dB (reg 0.1).
  - CLEAN already removes strong-target self-noise.
  - It is exact only for echo energy inside its own batch, so it cannot remove self-noise at long delays, and at full load it does not.
- **Fixes found by this study:**
  - α ≥ 0.5 normalisation could collapse a CPI by about 9 dB. The zero-weight floor was raised from 0.001 to 0.1 of mean energy.
  - The rebuilt reference had 2–4 false REs per CPI at idle load with a 12-bit ADC. Quantisation puts non-Gaussian energy into empty REs. This raised the ECA residual from 0.9 to 3.3 dB above thermal, and had made the reciprocal filter look 5 dB better than it is. Fixed with an absolute RE energy floor (0.004, about 5 dB below the weakest 256QAM point). LTE idle sensitivity improved by about 1 dB.
- **ADC:** a 12-bit ADC with gain set for busy-hour load adds about 0.9 dB of residual at idle (0 dB with 16 bits). Use ≥ 14 bits, or more direct-path rejection on the surveillance antenna.
- **Pd 0.9 range for a −10 dBsm drone with S5** (free space, sqrt(Rt·Rr)):

  | | idle | medium | full |
  |---|---|---|---|
  | LTE | 12.4 km | 15.8 km | 20.7 km |
  | NR | 7.1 km | 15.2 km | 20.6 km |

  Car-size targets are limited by the 20 km window and the radio horizon, not by SNR.
- **Open issue, idle load:** targets between the CFAR threshold and the 18 dB CLEAN threshold still produce comb ghosts. For NR this is the 50 Hz SSB comb (PSLR −1.2 dB); for LTE, the CRS comb. Fix candidates are a lower CLEAN threshold at sparse load, or AF-based ghost gating.

## Lessons already built into the code

0. **Estimate echoes after ECA, but subtract them before it.** ECA fitted with targets present removes each echo's zero-Doppler projection. That leaves a negative copy of the target at 0 Hz, which carries the NR 50 Hz SSB comb (ghosts at exactly ±50.0 Hz). CLEAN therefore subtracts echoes from the raw surveillance and re-runs ECA. `EcaCanceller` caches the reference-only Gram factorisation, and its whole-CPI path runs as two FFT convolutions, so re-runs are cheap.

1. **Reference fidelity must beat the surveillance DNR.** Remod errors leak into the ECA output at (error energy / signal energy) × DNR. For example, 42 falsely "active" REs out of millions (−38 dB) left +12 dB of uncancelled direct path. Mitigations already in place:
   - activity detection per PRB-symbol block, plus a per-RE noise test
   - a `structural_empty` mask for SSB/PSS guard REs
   - regression tests that require a bit-exact remod grid through the full receiver model
2. **The ECA needs pre-cursor taps** (`n_lead`). Band-limited sub-sample clutter delays have tails at negative lags; without them the residual stayed ~5 dB above noise.
3. **The channel estimate must be a delay-domain fit limited to the CP length.** A frequency-domain smoother biased the frequency-selective reference channel (EVM −49 dB instead of ≈ −75 dB).
4. **ADC gain should be set for busy-hour load,** not by a per-record AGC. Bursty idle signals otherwise clip.

## Known simplifications

- **Content and placement:** PBCH, PDCCH and PDSCH payloads are random (correct REs and modulation, no channel coding or scrambling). The NR SSB position ignores the sync raster, CORESET REG mapping is random, and point A is assumed at the carrier edge.
- **Demod-remod uses oracle modulation orders.** Blind per-PRB classification (or DCI decoding) is still to do.
- **Channel model:** static reference channel, no range migration, point clutter with zero Doppler (no internal motion or vegetation spread).
- **Load presets** match the 3GPP test models in occupancy and modulation, not in exact per-PRB power boosting.
- **Validation status:**
  - *NR:* cross-checked against [py3gpp](https://github.com/catkira/py3gpp), an open-source port of the MATLAB 5G Toolbox, in `tests/test_against_py3gpp.py` (93 checks). Coverage: Gold sequence, PSS/SSS and PBCH DMRS for 8 cell IDs × 4 SSB indices, SSB positions and values inside generated grids, PDSCH DMRS sequence and positions, and OFDM modulation (CP, mapping and timing match `nrOFDMModulate` to 1e-9). Not covered, because py3gpp has no generator for them: TRS (CSI-RS) and PDCCH DMRS.
  - *LTE:* cross-checked against srsRAN 4G's PHY library in `tests/test_against_srsran.py` (28 checks: 7 cell IDs × 1/2 CRS ports). Vectors come from `tools/srsran/build_and_dump.sh` (WSL), pinned in `tests/data/srsran/SOURCE.txt`. CRS and SSS are bit-exact in position and value. PSS positions match; values agree to 2.3e-4 rad phase, which is srsRAN's float32 Zadoff-Chu rounding. pr450's PSS also matches the exact integer-reduced phase to 1e-12. Not covered: PBCH/PDCCH/PDSCH, because pr450 models their content statistically.
  - `matlab/validate_pr450.m` (LTE Toolbox + 5G Toolbox) is kept as an optional third reference.

## Next steps (plan Phase 4/5)

- **Sidelobe and CLEAN tuning:**
  - Taylor or Chebyshev windows and α, chosen by Pd/Pfa rather than false-alarm count
  - AF-based ghost gating for the residual comb ghosts
  - a faster CLEAN fit (Toeplitz Gram or GPU)
- **Pd/Pfa Monte Carlo** against RCS and range for each load.
- **Load-profile time series** (diurnal) and ECA-B / ECA-S tuning.
- **Blind modulation classification** for remod; a sensitivity sweep over reference SNR.
- **Tracker** (Kalman in bistatic range/Doppler), then multistatic localisation.
- **Swap configs for real values** once the network's ARFCN, PCI, SSB/TRS configuration, sites and EIRP are known, or once a site survey decodes MIB/SIB1.
