"use strict";

/* pr450 Passive Radar Explorer — static front end for data exported by scripts/build_web_data.py */

const LOAD_LABEL = { idle: "Idle", low: "Low", medium: "Medium", high: "High", full: "Full" };
const LOAD_HINT = {
  idle: "Idle: only always-on signals (sync, broadcast, pilots) — the hardest case for a passive radar.",
  low: "Low: about 10 % of resource blocks carry user data, in bursts (16QAM).",
  medium: "Medium: about 30 % of resource blocks (64QAM).",
  high: "High: about 70 % of resource blocks (64QAM).",
  full: "Full: every resource block in every subframe (3GPP test model 1.1, QPSK).",
};
const STD_LABEL = { LTE: "LTE B72", NR: "5G NR n72" };
const RE_TYPES = [
  { name: "Empty", var: "--re-empty" },
  { name: "Pilot", var: "--re-pilot", nr: "Pilot (DMRS / TRS / PBCH-DMRS)", lte: "Pilot (CRS, always on)" },
  { name: "Sync", var: "--re-sync", nr: "Sync (PSS / SSS in SSB)", lte: "Sync (PSS / SSS)" },
  { name: "Control", var: "--re-control", nr: "Control (PBCH / PDCCH)", lte: "Control (PBCH / PDCCH)" },
  { name: "Data", var: "--re-data", nr: "Data (PDSCH)", lte: "Data (PDSCH)" },
];
const STAGES = [
  { key: "raw", step: "Step 1", title: "Raw", desc: "Correlate surveillance with reference" },
  { key: "eca", step: "Step 2", title: "ECA", desc: "Cancel direct path and clutter" },
  { key: "sidelobe", step: "Step 3", title: "Sidelobe control", desc: "Taylor windows in range and Doppler" },
  { key: "clean", step: "Step 4", title: "CLEAN", desc: "Subtract targets one by one, re-run ECA" },
];
const TARGET_COLORS = ["#e4572e", "#a259ff", "#17a2b8", "#f0a202", "#3fa37a"];
const RCS_CLASSES = [
  { key: "drone", label: "Drone −10 dBsm" },
  { key: "car", label: "Car +10 dBsm" },
  { key: "large_vehicle", label: "Truck +20 dBsm" },
];
const MOD_NAME = { 2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM" };

const state = { tab: "frames", std: "LTE", load: "medium", stage: "clean", rcs: "drone" };
const cache = new Map();
let index = null;

/* ------------------------------------------------------------------ utils */

const $ = (sel) => document.querySelector(sel);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

function b64u8(s) {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function getJSON(name) {
  if (!cache.has(name)) {
    cache.set(name, fetch(`data/${name}`).then((r) => {
      if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
      return r.json();
    }));
  }
  return cache.get(name);
}

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const k of Object.keys(state)) if (p.has(k)) state[k] = p.get(k);
}
function writeHash() {
  const p = new URLSearchParams(state);
  history.replaceState(null, "", `#${p.toString()}`);
}

const PLOT_CONFIG = {
  responsive: true, displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines", "hoverCompareCartesian"],
};

function axis(title, extra = {}) {
  return Object.assign({
    title: { text: title, font: { size: 12, color: css("--text-2") }, standoff: 8 },
    gridcolor: css("--grid"), zerolinecolor: css("--grid"), linecolor: css("--border"),
    tickfont: { color: css("--text-2"), size: 11 }, automargin: true,
  }, extra);
}

function layout(extra = {}) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, system-ui, sans-serif", size: 12, color: css("--text-2") },
    margin: { l: 56, r: 16, t: 8, b: 44 }, showlegend: false,
    hoverlabel: { font: { family: "Inter, system-ui, sans-serif", size: 12 } },
  }, extra);
}

function plot(id, data, lay) {
  return Plotly.react(id, data, lay, PLOT_CONFIG);
}

function kpiCard({ label, value, unit = "", note = "", tone = "" }) {
  return `<div class="kpi ${tone}"><div class="k-label">${label}</div>
    <div class="k-value">${value}<span class="k-unit">${unit}</span></div>
    <div class="k-note">${note}</div></div>`;
}

function median(arr) {
  const a = [...arr].sort((p, q) => p - q);
  return a.length ? a[Math.floor(a.length / 2)] : 0;
}

/* Peaks of a cut that stand clearly above its own random floor: structural ghosts, not noise. */
function localPeaks(x, y, { exclude = () => false, minDb = -45, n = 3, minSep = 0, aboveFloorDb = 15 } = {}) {
  const floor = median(y.filter((_, i) => !exclude(x[i])));
  minDb = Math.max(minDb, floor + aboveFloorDb);
  const peaks = [];
  for (let i = 1; i < y.length - 1; i++) {
    if (y[i] >= y[i - 1] && y[i] >= y[i + 1] && y[i] > minDb && !exclude(x[i])) peaks.push({ x: x[i], y: y[i] });
  }
  peaks.sort((a, b) => b.y - a.y);
  const out = [];
  for (const p of peaks) {
    if (out.every((q) => Math.abs(q.x - p.x) > minSep)) out.push(p);
    if (out.length === n) break;
  }
  return out;
}

function explainRange(km, std) {
  if (Math.abs(km - 20.0) < 0.4) return "cyclic-prefix replica";
  if (std === "LTE" && (Math.abs(km - 6.66) < 0.3 || Math.abs(km - 13.3) < 0.3)) return "CRS comb";
  if (std === "NR" && (Math.abs(km % 5.0) < 0.3 || Math.abs((km % 5.0) - 5.0) < 0.3)) return "pilot comb";
  return "";
}
function explainDoppler(hz, std) {
  const a = Math.abs(hz);
  if (std === "LTE" && Math.abs(a % 200) < 4) return "PSS/SSS every 5 ms";
  if (Math.abs(a % 50) < 3 || Math.abs((a % 50) - 50) < 3) return std === "NR" ? "SSB every 20 ms" : "20 ms periodicity";
  if (Math.abs(a % 100) < 3) return "10 ms periodicity";
  return "";
}

/* --------------------------------------------------------------- controls */

function buildSegmented(el, items, key, onChange) {
  el.innerHTML = items.map(([v, label]) =>
    `<button type="button" data-v="${v}" aria-pressed="${state[key] === v}">${label}</button>`).join("");
  el.onclick = (e) => {
    const b = e.target.closest("button");
    if (!b || state[key] === b.dataset.v) return;
    state[key] = b.dataset.v;
    el.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x.dataset.v === state[key]));
    onChange();
  };
}

function syncControls() {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === state.tab));
  $("#tab-frames").hidden = state.tab !== "frames";
  $("#tab-radar").hidden = state.tab !== "radar";
  $("#load-hint").textContent = LOAD_HINT[state.load];
}

/* ----------------------------------------------------------- tab 1: frames */

async function renderFrames() {
  const d = await getJSON(`frames_${state.std.toLowerCase()}_${state.load}.json`);
  const k = d.kpi;
  const std = d.std;

  $("#frame-kpis").innerHTML = [
    kpiCard({ label: "Occupied resource elements", value: fmt(k.occupied_pct), unit: "%", note: "share of the grid carrying anything" }),
    kpiCard({ label: "Pilot resource elements", value: fmt(k.pilot_pct, 1), unit: "%",
      note: std === "LTE" ? "CRS — transmitted even when idle" : "DMRS only where data is scheduled" }),
    kpiCard({ label: "Known energy", value: fmt(k.known_energy_pct), unit: "%", note: "pilots + sync: known without decoding" }),
    kpiCard({ label: "Mean power", value: fmt(k.mean_power_db), unit: "dB", note: "relative to a fully loaded carrier" }),
    kpiCard({ label: "Peak sidelobe (PSLR)", value: fmt(d.ambiguity.pslr_db), unit: "dB",
      note: "strongest ghost a target creates", tone: d.ambiguity.pslr_db > -10 ? "bad" : d.ambiguity.pslr_db > -15 ? "warn" : "good" }),
  ].join("");

  // resource grid
  const { n_sym, n_sc, t_ms, f_mhz } = d.grid;
  const u8 = b64u8(d.grid.re_type);
  const z = [], text = [];
  const shares = [0, 0, 0, 0, 0];
  for (let s = 0; s < n_sc; s++) {
    const row = new Array(n_sym), trow = new Array(n_sym);
    for (let l = 0; l < n_sym; l++) {
      const v = u8[l * n_sc + s];
      row[l] = v; trow[l] = RE_TYPES[v].name; shares[v]++;
    }
    z.push(row); text.push(trow);
  }
  const total = n_sym * n_sc;
  $("#re-legend").innerHTML = RE_TYPES.map((t, i) =>
    `<span><i style="background:${css(t.var)}"></i>${std === "LTE" ? (t.lte || t.name) : (t.nr || t.name)} <b>${fmt(100 * shares[i] / total, 1)} %</b></span>`).join("");
  $("#grid-sub").textContent = `${STD_LABEL[std]} · ${LOAD_LABEL[d.load]} load · 20 ms · ${n_sc} subcarriers × ${n_sym} OFDM symbols`;
  const cs = [];
  RE_TYPES.forEach((t, i) => { cs.push([i / 5, css(t.var)], [(i + 1) / 5, css(t.var)]); });
  const sfLines = [];
  for (let ms = 1; ms < 20; ms++) {
    sfLines.push({ type: "line", x0: ms, x1: ms, y0: f_mhz[0], y1: f_mhz[f_mhz.length - 1],
      line: { color: css("--border"), width: ms % 10 === 0 ? 1.6 : 0.6 } });
  }
  plot("plot-grid", [{
    type: "heatmap", x: t_ms, y: f_mhz, z, text, zmin: -0.5, zmax: 4.5, colorscale: cs, showscale: false,
    hovertemplate: "%{text}<br>t = %{x:.3f} ms<br>f = %{y:.3f} MHz<extra></extra>",
  }], layout({
    xaxis: axis("time [ms] — thin lines: 1 ms subframes, thick: 10 ms frame", { range: [0, 20], dtick: 1 }),
    yaxis: axis("frequency offset [MHz]"), shapes: sfLines, margin: { l: 60, r: 10, t: 6, b: 46 },
  }));

  // power per symbol
  plot("plot-power", [{
    type: "bar", x: t_ms, y: d.sym_power_db.map((v) => Math.max(v, -40) + 40), base: -40,
    marker: { color: css("--accent") }, width: 0.06,
    hovertemplate: "t = %{x:.2f} ms<br>%{customdata:.1f} dB<extra></extra>", customdata: d.sym_power_db,
  }], layout({ xaxis: axis("time [ms]", { range: [0, 20], dtick: 2 }), yaxis: axis("power [dB]", { range: [-30, 2] }), bargap: 0 }));

  // spectrum
  plot("plot-spectrum", [{
    type: "scatter", mode: "lines", x: d.spectrum.f_mhz, y: d.spectrum.db, line: { color: css("--accent"), width: 1.4 },
    hovertemplate: "%{x:.2f} MHz<br>%{y:.1f} dB<extra></extra>",
  }], layout({
    xaxis: axis("frequency offset [MHz]", { range: [-3.84, 3.84] }), yaxis: axis("PSD [dB]", { range: [-45, 3] }),
    shapes: [{ type: "rect", x0: -2.25, x1: 2.25, y0: -45, y1: 3, fillcolor: css("--accent-soft"), line: { width: 0 }, layer: "below" }],
  }));

  // ambiguity cuts
  const af = d.ambiguity;
  const rPeaks = localPeaks(af.range_km, af.range_cut_db, { exclude: (x) => x < 1.5, minDb: -40, n: 3, minSep: 1.0 });
  const dPeaks = localPeaks(af.doppler_hz, af.doppler_cut_db, { exclude: (x) => Math.abs(x) < 12 || x < 0, minDb: -40, n: 3, minSep: 20 });
  const rFloor = median(af.range_cut_db.filter((_, i) => af.range_km[i] >= 1.5));
  const dFloor = median(af.doppler_cut_db.filter((_, i) => Math.abs(af.doppler_hz[i]) >= 12));
  const annot = (peaks, unit, explain) => [...peaks].sort((a, b) => a.x - b.x).map((p, i) => ({
    x: p.x, y: p.y,
    text: `${fmt(p.x, unit === "km" ? 1 : 0)} ${unit} · ${fmt(p.y)} dB${explain(p.x) ? "<br>" + explain(p.x) : ""}`,
    showarrow: true, arrowhead: 0, arrowcolor: css("--text-3"), ax: 0, ay: -24 - 34 * (i % 2),
    font: { size: 10.5, color: css("--text") }, bgcolor: css("--surface"), bordercolor: css("--border"), borderpad: 2,
  }));
  plot("plot-af-range", [{
    type: "scatter", mode: "lines", x: af.range_km, y: af.range_cut_db, line: { color: css("--re-pilot"), width: 1.2 },
    hovertemplate: "%{x:.2f} km<br>%{y:.1f} dB<extra></extra>",
  }], layout({ xaxis: axis("bistatic range [km]"), yaxis: axis("dB rel. peak", { range: [-65, 6] }),
    annotations: annot(rPeaks, "km", (x) => explainRange(x, std)) }));
  plot("plot-af-doppler", [{
    type: "scatter", mode: "lines", x: af.doppler_hz, y: af.doppler_cut_db, line: { color: css("--re-sync"), width: 1.2 },
    hovertemplate: "%{x:.0f} Hz<br>%{y:.1f} dB<extra></extra>",
  }], layout({ xaxis: axis("Doppler [Hz]"), yaxis: axis("dB rel. peak", { range: [-65, 6] }),
    annotations: annot(dPeaks, "Hz", (x) => explainDoppler(x, std)) }));
  const top = rPeaks[0];
  $("#af-range-caption").textContent = top
    ? `Strongest range ghost: ${fmt(top.y)} dB at ${fmt(top.x)} km${explainRange(top.x, std) ? " (" + explainRange(top.x, std) + ")" : ""}. A target 60 dB above noise would leave a ghost ${fmt(60 + top.y, 0)} dB above noise.`
    : `No range ghosts: only the random data floor (median ${fmt(rFloor, 0)} dB).`;
  const topD = dPeaks[0];
  $("#af-doppler-caption").textContent = topD
    ? `Strongest Doppler ghost: ${fmt(topD.y)} dB at ±${fmt(topD.x, 0)} Hz${explainDoppler(topD.x, std) ? " (" + explainDoppler(topD.x, std) + ")" : ""}.`
    : `No periodic Doppler structure: traffic randomises the signal (floor median ${fmt(dFloor, 0)} dB).`;

  // constellation
  const mods = d.constellation.data_bits.map((b) => MOD_NAME[b]).filter(Boolean);
  $("#const-sub").textContent = d.constellation.i.length
    ? `${d.constellation.i.length} random payload REs · data: ${mods.join(", ") || "none"} · control: QPSK`
    : "no payload at this load";
  plot("plot-const", [{
    type: "scattergl", mode: "markers", x: d.constellation.i, y: d.constellation.q,
    marker: { size: 5, color: css("--re-data"), opacity: 0.75 }, hoverinfo: "skip",
  }], layout({ xaxis: axis("I", { range: [-1.6, 1.6], zeroline: true }),
    yaxis: axis("Q", { range: [-1.6, 1.6], scaleanchor: "x", zeroline: true }), margin: { l: 48, r: 16, t: 8, b: 40 } }));
}

/* ------------------------------------------------------------ tab 2: radar */

function stageCaption(d, s) {
  const st = d.stages[s];
  const k = d.kpi;
  switch (s) {
    case "raw":
      return `Without cancellation the direct path (${fmt(k.surv_dnr_db, 0)} dB above noise) and clutter dominate the map near 0 km and 0 Hz; their sidelobes bury the targets (${st.hits} of 3 found).`;
    case "eca":
      return `ECA removes ${fmt(k.eca_cancellation_db, 0)} dB of direct path and clutter. All targets appear, but each drags sidelobes and comb ghosts along: ${st.false_alarms} false alarms.`;
    case "sidelobe":
      return `Taylor windows lower range and Doppler sidelobes (${st.false_alarms} false alarms left). Periodic-signal comb ghosts are untouched by windows.`;
    default:
      return `CLEAN subtracts each detected target (◆, ${st.clean_components} components) and re-runs ECA, so its ghosts vanish with it. The map shows the residual: ${st.false_alarms} false alarm${st.false_alarms === 1 ? "" : "s"}.`;
  }
}

async function renderRadar() {
  const d = await getJSON(`radar_${state.std.toLowerCase()}_${state.load}.json`);
  const s = state.stage;
  const st = d.stages[s];
  const k = d.kpi;

  // pipeline stepper
  $("#pipeline").innerHTML = STAGES.map((g) => {
    const x = d.stages[g.key];
    return `<button type="button" class="stage" data-stage="${g.key}" aria-pressed="${g.key === s}">
      <div class="s-step">${g.step}</div><div class="s-title">${g.title}</div><div class="s-desc">${g.desc}</div>
      <div class="s-stats"><span>targets <b>${x.hits}/3</b></span><span>false alarms <b>${x.false_alarms}</b></span></div></button>`;
  }).join("");

  const faTone = st.false_alarms <= 2 ? "good" : st.false_alarms <= 20 ? "warn" : "bad";
  $("#radar-kpis").innerHTML = [
    kpiCard({ label: "Direct path in surveillance", value: fmt(k.surv_dnr_db), unit: "dB", note: "above receiver noise" }),
    kpiCard({ label: "Strongest clutter", value: fmt(k.clutter_cnr_db), unit: "dB", note: "static scatterers near the receiver" }),
    kpiCard({ label: "Reference rebuild", value: k.remod_wrong_res, unit: "wrong REs", note: "demod-remod of the reference channel",
      tone: k.remod_wrong_res === 0 ? "good" : "bad" }),
    kpiCard({ label: "ECA cancellation", value: fmt(k.eca_cancellation_db), unit: "dB", note: `${k.eca_taps} taps, whole ${k.cpi_s} s CPI` }),
    kpiCard({ label: "Targets detected", value: `${st.hits}/3`, note: `${STAGES.find((g) => g.key === s).title} stage`,
      tone: st.hits === 3 ? "good" : "bad" }),
    kpiCard({ label: "False alarms", value: st.false_alarms, note: `Pfa 1e-6 over ${(k.cells / 1000).toFixed(0)}k cells (≈0.1 from noise)`, tone: faTone }),
  ].join("");

  const g = d.geometry;
  const targets = g.targets;

  // scene
  const sceneData = [
    { type: "scatter", mode: "markers", name: "clutter", x: g.clutter.map((c) => c.x_km), y: g.clutter.map((c) => c.y_km),
      marker: { size: g.clutter.map((c) => 4 + Math.max(0, c.rcs - 20) / 4), color: css("--text-3"), opacity: 0.5 },
      hovertemplate: "clutter<br>%{customdata:.0f} dBsm<extra></extra>", customdata: g.clutter.map((c) => c.rcs) },
    { type: "scatter", mode: "lines", name: "baseline", x: [g.tx[0], g.rx[0]], y: [g.tx[1], g.rx[1]],
      line: { color: css("--text-3"), width: 1, dash: "dot" }, hoverinfo: "skip" },
    ...targets.map((t, i) => ({ type: "scatter", mode: "lines", name: `${t.name} ellipse`, x: t.ellipse.x, y: t.ellipse.y,
      line: { color: TARGET_COLORS[i], width: 1, dash: "dash" }, opacity: 0.55, hoverinfo: "skip" })),
    { type: "scatter", mode: "markers+text", name: "transmitter", x: [g.tx[0]], y: [g.tx[1]], text: ["Tx"], textposition: "middle left",
      marker: { symbol: "triangle-up", size: 15, color: css("--re-pilot") }, textfont: { color: css("--text-2"), size: 11 },
      hovertemplate: "transmitter (illuminator)<extra></extra>" },
    { type: "scatter", mode: "markers+text", name: "receiver", x: [g.rx[0]], y: [g.rx[1]], text: ["Rx"], textposition: "middle right",
      marker: { symbol: "square", size: 12, color: css("--accent") }, textfont: { color: css("--text-2"), size: 11 },
      hovertemplate: "passive receiver<extra></extra>" },
    { type: "scatter", mode: "markers+text", name: "targets", x: targets.map((t) => t.x_km), y: targets.map((t) => t.y_km),
      text: targets.map((t) => t.name), textposition: "top center",
      marker: { size: 11, color: targets.map((_, i) => TARGET_COLORS[i]), line: { color: css("--surface"), width: 1.5 } },
      textfont: { color: css("--text"), size: 11 },
      customdata: targets.map((t) => [t.speed_mps, t.rcs_dbsm, t.r_km]),
      hovertemplate: "%{text}<br>%{customdata[0]} m/s · %{customdata[1]} dBsm<br>bistatic range %{customdata[2]:.2f} km<extra></extra>" },
  ];
  const arrows = targets.map((t, i) => ({
    x: t.x_km + t.vx * 0.06, y: t.y_km + t.vy * 0.06, ax: t.x_km, ay: t.y_km, xref: "x", yref: "y", axref: "x", ayref: "y",
    showarrow: true, arrowhead: 2, arrowsize: 1, arrowwidth: 1.6, arrowcolor: TARGET_COLORS[i], text: "",
  }));
  plot("plot-scene", sceneData, layout({
    xaxis: axis("east [km]", { range: [-4, 12] }), yaxis: axis("north [km]", { scaleanchor: "x", range: [-8, 6] }),
    annotations: arrows, margin: { l: 50, r: 12, t: 8, b: 42 },
  }));

  // range-Doppler map
  const ax = d.axes;
  const u8 = b64u8(st.map);
  const z = [];
  for (let i = 0; i < ax.n_range; i++) {
    const row = new Array(ax.n_doppler);
    for (let j = 0; j < ax.n_doppler; j++) row[j] = u8[i * ax.n_doppler + j] * ax.db_step;
    z.push(row);
  }
  const zmax = s === "raw" ? 100 : 60;
  const nComp = st.clean_components;
  const comps = st.detections.slice(0, nComp);
  const dets = st.detections.slice(nComp);
  const rdData = [
    { type: "heatmap", x: ax.doppler_hz, y: ax.range_km, z, zmin: 0, zmax, colorscale: "Viridis",
      colorbar: { title: { text: "dB over noise", side: "right", font: { size: 11, color: css("--text-2") } }, thickness: 12,
        tickfont: { color: css("--text-2"), size: 11 }, outlinewidth: 0 },
      hovertemplate: "%{y:.2f} km · %{x:.1f} Hz<br>%{z:.1f} dB<extra></extra>" },
    { type: "scatter", mode: "markers", name: "detections", x: dets.map((p) => p.fd), y: dets.map((p) => p.r_km),
      marker: { symbol: "x-thin", size: 8, color: "#ffffff", line: { width: 1.6, color: "#ffffff" } },
      customdata: dets.map((p) => p.snr), hovertemplate: "CFAR detection<br>%{y:.2f} km · %{x:.1f} Hz<br>%{customdata:.1f} dB over local noise<extra></extra>" },
    { type: "scatter", mode: "markers", name: "CLEAN", x: comps.map((p) => p.fd), y: comps.map((p) => p.r_km),
      marker: { symbol: "diamond", size: 9, color: "#ffffff", line: { width: 1.5, color: "#111" } },
      hovertemplate: "removed by CLEAN<br>%{y:.2f} km · %{x:.2f} Hz<extra></extra>" },
    { type: "scatter", mode: "markers", name: "truth", x: targets.map((t) => t.fd), y: targets.map((t) => t.r_km),
      marker: { symbol: "circle-open", size: 20, color: targets.map((_, i) => TARGET_COLORS[i]), line: { width: 2.4 } },
      text: targets.map((t) => t.name), hovertemplate: "truth: %{text}<br>%{y:.2f} km · %{x:.1f} Hz<extra></extra>" },
  ];
  plot("plot-rd", rdData, layout({
    xaxis: axis("Doppler [Hz]", { range: [ax.doppler_hz[0], ax.doppler_hz[ax.doppler_hz.length - 1]] }),
    yaxis: axis("bistatic range [km]", { range: [0, ax.range_km[ax.range_km.length - 1]] }),
    margin: { l: 56, r: 8, t: 8, b: 44 },
  }));
  $("#rd-sub").textContent = `${STD_LABEL[d.std]} · ${LOAD_LABEL[d.load]} load · ${STAGES.find((x) => x.key === s).title} · ${k.cpi_s} s CPI`;
  $("#rd-caption").textContent = stageCaption(d, s);

  // target table
  const rows = targets.map((t, i) => {
    const snr = st.target_snr[i];
    return `<tr><td><span class="swatch" style="background:${TARGET_COLORS[i]}"></span>${t.name}</td>
      <td>${fmt(t.rcs_dbsm, 0)}</td><td>${fmt(t.speed_mps)}</td><td>${fmt(t.r_km, 2)}</td><td>${fmt(t.fd)}</td>
      <td>${fmt(t.snr_per_sample_db)}</td>
      <td>${snr === null ? '<span class="pill no">missed</span>' : `<span class="pill yes">detected</span> ${fmt(snr)} dB`}</td></tr>`;
  }).join("");
  $("#target-table").innerHTML = `<thead><tr><th>Target</th><th title="radar cross-section">RCS<br><small>dBsm</small></th>
    <th>Speed<br><small>m/s</small></th><th>Range<br><small>km bistatic</small></th><th>Doppler<br><small>Hz</small></th>
    <th title="echo power over receiver noise, per sample, before integration">Echo SNR<br><small>dB / sample</small></th>
    <th>${STAGES.find((x) => x.key === s).title}<br><small>map SNR</small></th></tr></thead><tbody>${rows}</tbody>`;

  renderRangeChart();
}

function renderRangeChart() {
  const tab = index && index.detection_range_km;
  if (!tab || !Object.keys(tab).length) {
    $("#plot-range").innerHTML = '<p class="caption">Run scripts/run_pd_pfa_study.py to add this chart.</p>';
    return;
  }
  const loads = ["idle", "medium", "full"];
  const series = ["LTE", "NR"].filter((sd) => tab[sd]).map((sd, n) => ({
    type: "bar", name: STD_LABEL[sd], x: loads.map((l) => LOAD_LABEL[l]),
    y: loads.map((l) => (tab[sd][l] ? tab[sd][l][state.rcs] : null)),
    marker: {
      color: n === 0 ? css("--re-pilot") : css("--re-sync"),
      opacity: loads.map((l) => (sd === state.std && l === state.load ? 1 : 0.45)),
      line: { width: loads.map((l) => (sd === state.std && l === state.load ? 2 : 0)), color: css("--text") },
    },
    text: loads.map((l) => (tab[sd][l] ? `${tab[sd][l][state.rcs].toFixed(1)}` : "")), textposition: "outside",
    textfont: { size: 11, color: css("--text-2") }, cliponaxis: false,
    hovertemplate: `${STD_LABEL[sd]} · %{x}<br>%{y:.1f} km<extra></extra>`,
  }));
  const ymax = Math.max(...series.flatMap((x) => x.y.filter((v) => v !== null))) * 1.18;
  plot("plot-range", series, layout({
    barmode: "group", showlegend: true, legend: { orientation: "h", y: 1.12, x: 0, font: { size: 11 } },
    xaxis: axis("cell load"), yaxis: axis("range [km]", { range: [0, ymax] }),
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 20, y1: 20, line: { color: css("--text-3"), dash: "dash", width: 1 } }],
    annotations: [{ xref: "paper", x: 0, y: 20, yanchor: "bottom", xanchor: "left", showarrow: false,
      text: "20 km processed window", font: { size: 10.5, color: css("--text-3") }, bgcolor: css("--surface") }],
    margin: { l: 50, r: 10, t: 26, b: 40 },
  }));
}

/* -------------------------------------------------------------- lifecycle */

let renderToken = 0;
async function render() {
  syncControls();
  writeHash();
  const token = ++renderToken;
  const main = $("main");
  main.classList.add("loading");
  try {
    if (state.tab === "frames") await renderFrames(); else await renderRadar();
  } catch (err) {
    console.error(err);
    const panel = state.tab === "frames" ? $("#frame-kpis") : $("#radar-kpis");
    panel.innerHTML = `<div class="kpi bad"><div class="k-label">Could not load data</div><div class="k-note">${err.message}. Run scripts/build_web_data.py.</div></div>`;
  } finally {
    if (token === renderToken) main.classList.remove("loading");
  }
}

async function init() {
  readHash();
  try { index = await getJSON("index.json"); } catch (e) { index = { loads: Object.keys(LOAD_LABEL), standards: STD_LABEL }; }
  const loads = (index.loads || Object.keys(LOAD_LABEL)).filter((l) => LOAD_LABEL[l]);
  if (!loads.includes(state.load)) state.load = loads.includes("medium") ? "medium" : loads[0];
  if (!STD_LABEL[state.std]) state.std = "LTE";
  if (!STAGES.some((g) => g.key === state.stage)) state.stage = "clean";

  buildSegmented($("#std-seg"), Object.keys(STD_LABEL).map((s) => [s, STD_LABEL[s]]), "std", render);
  buildSegmented($("#load-seg"), loads.map((l) => [l, LOAD_LABEL[l]]), "load", render);
  buildSegmented($("#rcs-seg"), RCS_CLASSES.map((c) => [c.key, c.label]), "rcs", renderRangeChart);
  document.querySelector(".tabs").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-tab]");
    if (b && b.dataset.tab !== state.tab) { state.tab = b.dataset.tab; render(); }
  });
  $("#pipeline").addEventListener("click", (e) => {
    const b = e.target.closest(".stage");
    if (b && b.dataset.stage !== state.stage) { state.stage = b.dataset.stage; render(); }
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  if (index.built) $("#build-info").textContent = `Data built ${index.built.replace("T", " ")} · pr450`;
  render();
}

document.addEventListener("DOMContentLoaded", init);
