# Pd / Pfa Monte Carlo study

1728 target trials per setting, 0.5 s CPI, CA-CFAR Pfa 1e-6 (N = 120), baseline clutter + 12-bit ADC. Ideal CA-CFAR needs a map SNR of 11.5 dB (Pd 0.5) / 13.4 dB (Pd 0.9) for a non-fluctuating target.

SNR = per-sample echo SNR over receiver noise for a fully loaded carrier (link-budget quantity).

| Std | Load | Setting | SNR for Pd 0.9 [dB] (90 % CI) | Processing loss [dB] | FA at 1e-6 (measured Pfa) | Pfa at nominal 1e-4 |
|---|---|---|---|---|---|---|
| LTE | idle | S0 | -40.0 (-40.7 to -39.4) | 3.5 | 67 (6.9e-05) | 2.6e-04 |
| LTE | idle | S1 | -35.4 (-36.2 to -34.8) | 8.2 | 4 (4.1e-06) | 9.0e-05 |
| LTE | idle | S2 | -35.8 (-36.5 to -35.3) | 7.8 | 5 (5.2e-06) | 1.0e-04 |
| LTE | idle | S3 | -39.1 (-39.6 to -38.5) | 4.5 | 40 (4.1e-05) | 2.0e-04 |
| LTE | idle | S4 | -38.2 (-38.7 to -37.7) | 5.4 | 16 (1.6e-05) | 1.7e-04 |
| LTE | idle | S5 | -39.4 (-40.0 to -38.7) | 4.2 | 38 (3.9e-05) | 2.0e-04 |
| LTE | idle | S6 | -39.1 (-39.8 to -38.5) | 4.5 | 27 (2.8e-05) | 1.9e-04 |
| LTE | idle | S7 | -35.8 (-36.4 to -35.2) | 7.8 | 4 (4.1e-06) | 1.6e-04 |
| LTE | medium | S0 | -44.7 (-45.5 to -44.1) | 3.7 | 0 (0.0e+00) | 1.1e-04 |
| LTE | medium | S1 | -40.6 (-41.4 to -40.0) | 7.7 | 0 (0.0e+00) | 9.2e-05 |
| LTE | medium | S2 | -40.6 (-41.5 to -39.8) | 7.8 | 1 (1.0e-06) | 8.8e-05 |
| LTE | medium | S3 | -43.7 (-44.6 to -42.9) | 4.6 | 1 (1.0e-06) | 1.2e-04 |
| LTE | medium | S4 | -43.2 (-43.8 to -42.7) | 5.1 | 1 (1.0e-06) | 1.1e-04 |
| LTE | medium | S5 | -43.6 (-44.6 to -42.8) | 4.8 | 1 (1.0e-06) | 1.2e-04 |
| LTE | medium | S6 | -43.3 (-44.0 to -42.8) | 5.0 | 0 (0.0e+00) | 1.5e-04 |
| LTE | medium | S7 | -40.4 (-41.2 to -39.7) | 8.0 | 1 (1.0e-06) | 1.3e-04 |
| LTE | full | S0 | -49.2 (-49.9 to -48.6) | 3.1 | 0 (0.0e+00) | 1.2e-04 |
| LTE | full | S1 | -44.9 (-45.5 to -44.3) | 7.3 | 4 (4.1e-06) | 8.4e-05 |
| LTE | full | S2 | -44.7 (-45.4 to -44.2) | 7.5 | 4 (4.1e-06) | 9.1e-05 |
| LTE | full | S3 | -48.3 (-49.0 to -47.8) | 3.9 | 0 (0.0e+00) | 1.1e-04 |
| LTE | full | S4 | -47.4 (-48.0 to -46.9) | 4.8 | 1 (1.0e-06) | 9.9e-05 |
| LTE | full | S5 | -48.3 (-49.0 to -47.8) | 3.9 | 0 (0.0e+00) | 1.1e-04 |
| LTE | full | S6 | -48.2 (-48.8 to -47.8) | 4.0 | 1 (1.0e-06) | 1.1e-04 |
| LTE | full | S7 | -46.5 (-47.1 to -46.0) | 5.7 | 2 (2.1e-06) | 1.1e-04 |
| NR | idle | S0 | -30.1 (-31.0 to -29.4) | 5.7 | 287 (3.0e-04) | 5.6e-04 |
| NR | idle | S1 | -26.3 (-27.6 to -24.8) | 9.5 | 91 (9.4e-05) | 2.7e-04 |
| NR | idle | S2 | -26.6 (-27.8 to -25.4) | 9.2 | 129 (1.3e-04) | 3.4e-04 |
| NR | idle | S3 | -29.4 (-30.3 to -28.6) | 6.4 | 249 (2.6e-04) | 5.1e-04 |
| NR | idle | S4 | -28.8 (-29.7 to -27.9) | 7.1 | 182 (1.9e-04) | 4.0e-04 |
| NR | idle | S5 | -29.8 (-30.6 to -28.9) | 6.1 | 281 (2.9e-04) | 5.7e-04 |
| NR | idle | S6 | -28.5 (-29.5 to -27.5) | 7.4 | 288 (3.0e-04) | 6.0e-04 |
| NR | idle | S7 | -25.4 (-26.7 to -23.8) | 10.4 | 152 (1.6e-04) | 4.2e-04 |
| NR | medium | S0 | -43.1 (-43.7 to -42.5) | 4.1 | 0 (0.0e+00) | 1.1e-04 |
| NR | medium | S1 | -40.0 (-40.8 to -39.4) | 7.1 | 5 (5.2e-06) | 9.9e-05 |
| NR | medium | S2 | -40.0 (-40.8 to -39.4) | 7.1 | 4 (4.1e-06) | 9.3e-05 |
| NR | medium | S3 | -42.8 (-43.3 to -42.3) | 4.4 | 2 (2.1e-06) | 1.2e-04 |
| NR | medium | S4 | -42.2 (-42.7 to -41.7) | 5.0 | 2 (2.1e-06) | 1.3e-04 |
| NR | medium | S5 | -42.9 (-43.4 to -42.4) | 4.2 | 2 (2.1e-06) | 1.3e-04 |
| NR | medium | S6 | -42.4 (-43.1 to -41.8) | 4.8 | 3 (3.1e-06) | 1.3e-04 |
| NR | medium | S7 | -39.3 (-40.0 to -38.6) | 7.9 | 5 (5.2e-06) | 1.3e-04 |
| NR | full | S0 | -48.6 (-49.4 to -48.0) | 3.4 | 2 (2.1e-06) | 1.0e-04 |
| NR | full | S1 | -45.4 (-46.1 to -44.8) | 6.6 | 6 (6.2e-06) | 9.3e-05 |
| NR | full | S2 | -45.6 (-46.2 to -45.0) | 6.4 | 9 (9.3e-06) | 8.9e-05 |
| NR | full | S3 | -48.2 (-48.8 to -47.8) | 3.8 | 5 (5.2e-06) | 1.2e-04 |
| NR | full | S4 | -47.3 (-47.9 to -46.9) | 4.6 | 6 (6.2e-06) | 1.1e-04 |
| NR | full | S5 | -48.2 (-48.8 to -47.8) | 3.7 | 5 (5.2e-06) | 1.2e-04 |
| NR | full | S6 | -48.0 (-48.5 to -47.6) | 4.0 | 4 (4.1e-06) | 1.3e-04 |
| NR | full | S7 | -46.7 (-47.2 to -46.3) | 5.3 | 2 (2.1e-06) | 1.3e-04 |

Settings:
- S0: Hann Doppler, no range taper, alpha 0
- S1: Blackman-Harris both, alpha 0.5 (current; normalisation floor 0.1)
- S2: Blackman-Harris both, alpha 0
- S3: Taylor -40 dB both, alpha 0.25
- S4: Hann both, alpha 0.5
- S5: Taylor -40 dB both, alpha 0 (adopted)
- S6: S5 + batch reciprocal filter, reg 1
- S7: S5 + batch reciprocal filter, reg 0.1

## Detection range, Pd 0.9 (free space, baseline geometry: EIRP 61 dBm, 10 dBi, NF 5 dB, 3 km baseline)

Geometric-mean range sqrt(Rt Rr) [km] / range behind the receiver on the baseline axis [km]. RCS values are assumptions.

These are free-space link-budget limits, not coverage. In practice they are capped by: the processed window (20 km bistatic range, 512 delay bins, configurable); radio line of sight (4/3-Earth: about 28 km from a 30 m Tx and 19 km from a 10 m Rx to a 2 m ground target, about 54 km from the Rx to a drone at 100 m); and terrain and propagation losses, which are not modelled.

| Std | Load | Setting | small drone (-10 dBsm) | car (+10 dBsm) | large vehicle (+20 dBsm) |
|---|---|---|---|---|---|
| LTE | idle | S0 | 12.9 / 11.5 | 40.8 / 39.3 | 72.5 / 71.0 |
| LTE | idle | S1 | 9.9 / 8.5 | 31.2 / 29.7 | 55.5 / 54.0 |
| LTE | idle | S2 | 10.1 / 8.7 | 31.9 / 30.5 | 56.8 / 55.3 |
| LTE | idle | S3 | 12.2 / 10.8 | 38.5 / 37.0 | 68.5 / 67.0 |
| LTE | idle | S4 | 11.6 / 10.2 | 36.6 / 35.1 | 65.1 / 63.6 |
| LTE | idle | S5 | 12.4 / 11.0 | 39.3 / 37.8 | 69.8 / 68.3 |
| LTE | idle | S6 | 12.2 / 10.8 | 38.6 / 37.1 | 68.6 / 67.2 |
| LTE | idle | S7 | 10.1 / 8.7 | 31.9 / 30.5 | 56.8 / 55.3 |
| LTE | medium | S0 | 16.8 / 15.4 | 53.2 / 51.7 | 94.6 / 93.2 |
| LTE | medium | S1 | 13.3 / 11.9 | 42.2 / 40.7 | 75.0 / 73.5 |
| LTE | medium | S2 | 13.3 / 11.9 | 42.0 / 40.5 | 74.7 / 73.2 |
| LTE | medium | S3 | 15.9 / 14.5 | 50.4 / 48.9 | 89.6 / 88.1 |
| LTE | medium | S4 | 15.5 / 14.1 | 49.0 / 47.5 | 87.1 / 85.6 |
| LTE | medium | S5 | 15.8 / 14.4 | 50.0 / 48.6 | 89.0 / 87.5 |
| LTE | medium | S6 | 15.6 / 14.1 | 49.2 / 47.8 | 87.6 / 86.1 |
| LTE | medium | S7 | 13.2 / 11.8 | 41.6 / 40.2 | 74.1 / 72.6 |
| LTE | full | S0 | 21.8 / 20.3 | 68.9 / 67.4 | 122.4 / 121.0 |
| LTE | full | S1 | 17.0 / 15.6 | 53.9 / 52.4 | 95.8 / 94.3 |
| LTE | full | S2 | 16.9 / 15.4 | 53.3 / 51.8 | 94.8 / 93.3 |
| LTE | full | S3 | 20.7 / 19.3 | 65.5 / 64.0 | 116.5 / 115.0 |
| LTE | full | S4 | 19.7 / 18.2 | 62.2 / 60.8 | 110.7 / 109.2 |
| LTE | full | S5 | 20.7 / 19.3 | 65.5 / 64.0 | 116.5 / 115.0 |
| LTE | full | S6 | 20.6 / 19.2 | 65.3 / 63.8 | 116.1 / 114.6 |
| LTE | full | S7 | 18.7 / 17.2 | 59.1 / 57.6 | 105.0 / 103.5 |
| NR | idle | S0 | 7.3 / 5.9 | 23.0 / 21.6 | 40.9 / 39.5 |
| NR | idle | S1 | 5.8 / 4.5 | 18.5 / 17.0 | 32.9 / 31.4 |
| NR | idle | S2 | 5.9 / 4.6 | 18.8 / 17.3 | 33.4 / 31.9 |
| NR | idle | S3 | 7.0 / 5.7 | 22.1 / 20.7 | 39.4 / 37.9 |
| NR | idle | S4 | 6.7 / 5.4 | 21.3 / 19.9 | 37.9 / 36.4 |
| NR | idle | S5 | 7.1 / 5.8 | 22.5 / 21.1 | 40.1 / 38.6 |
| NR | idle | S6 | 6.6 / 5.3 | 20.9 / 19.5 | 37.2 / 35.8 |
| NR | idle | S7 | 5.6 / 4.3 | 17.6 / 16.1 | 31.3 / 29.8 |
| NR | medium | S0 | 15.4 / 13.9 | 48.6 / 47.1 | 86.4 / 84.9 |
| NR | medium | S1 | 12.9 / 11.5 | 40.8 / 39.3 | 72.5 / 71.0 |
| NR | medium | S2 | 12.9 / 11.5 | 40.8 / 39.3 | 72.5 / 71.0 |
| NR | medium | S3 | 15.1 / 13.7 | 47.7 / 46.2 | 84.9 / 83.4 |
| NR | medium | S4 | 14.6 / 13.2 | 46.1 / 44.7 | 82.0 / 80.6 |
| NR | medium | S5 | 15.2 / 13.8 | 48.1 / 46.6 | 85.5 / 84.0 |
| NR | medium | S6 | 14.7 / 13.3 | 46.6 / 45.2 | 82.9 / 81.4 |
| NR | medium | S7 | 12.3 / 10.9 | 39.0 / 37.5 | 69.4 / 67.9 |
| NR | full | S0 | 21.1 / 19.6 | 66.7 / 65.2 | 118.5 / 117.0 |
| NR | full | S1 | 17.5 / 16.1 | 55.4 / 53.9 | 98.6 / 97.1 |
| NR | full | S2 | 17.7 / 16.3 | 56.0 / 54.5 | 99.5 / 98.0 |
| NR | full | S3 | 20.6 / 19.2 | 65.2 / 63.7 | 115.9 / 114.4 |
| NR | full | S4 | 19.6 / 18.2 | 62.0 / 60.5 | 110.3 / 108.8 |
| NR | full | S5 | 20.6 / 19.2 | 65.3 / 63.8 | 116.1 / 114.6 |
| NR | full | S6 | 20.3 / 18.9 | 64.3 / 62.9 | 114.4 / 112.9 |
| NR | full | S7 | 18.9 / 17.4 | 59.7 / 58.2 | 106.2 / 104.7 |
