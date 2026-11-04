# pr450 Passive Radar Explorer (static web app)

Two tabs, driven by data exported from the pr450 Python pipeline:

1. **OFDM frames**: resource grid (20 ms), power per symbol, spectrum, self-ambiguity cuts with ghost labels, payload constellation, for LTE B72 / NR n72 at idle / low / medium / high / full load.
2. **Passive radar**: scene geometry, range-Doppler map per processing stage (raw → ECA → sidelobe control → CLEAN), KPIs, target table and detection range from the Pd/Pfa study.

## Rebuild the data

```bash
python scripts/build_web_data.py          # ~3-4 min, writes web/data/ (~8 MB)
python scripts/run_pd_pfa_study.py        # optional: refreshes the detection-range chart (~9 min)
```

## View locally

```bash
python -m http.server 8740 --directory web
```

Then open http://localhost:8740. Opening `index.html` as a file doesn't work, because browsers block `fetch` from `file://`.

## Publish on GitHub Pages

Everything under `web/` is static (HTML, CSS, JS, JSON). Plotly loads from jsDelivr and fonts from Google Fonts.

- **Project repository:** push the repo, then set Settings → Pages → Source to **GitHub Actions**. `.github/workflows/pages.yml` publishes `web/` on every push to `main`.
- **Existing user site (`<user>.github.io`):** copy the contents of `web/` into a sub-folder of that site, e.g. `/pr450/`. All paths are relative.
