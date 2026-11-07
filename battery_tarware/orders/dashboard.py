"""Offline, self-contained HTML replay dashboard (no server, no external libraries).

    python -m battery_tarware.orders.dashboard trace_fifo.json trace_plb.json -o dashboard.html

Open it in a browser; ``dashboard.html#run=1&t=300&order=O00012`` opens run 1 at tick 300 with an
order selected (also ``task=``, ``robot=``, ``pkg=``, ``station=15,10``, ``charger=0``).

or ``write_dashboard([trace, ...], "dashboard.html")`` with traces from ``run_episode``. One file
holds the recorded runs; everything shown is reconstructed from each run's single trace (event log
and per-tick frames), so the dashboard, CSV/JSON exports and metrics agree.

Panels, synchronised on the selected tick: warehouse map (zones, shelves, stations, charging docks,
robots with battery, carried shelves, selected crew), order-task **kanban** (one column per lifecycle
state), order flow (cumulative arrived / completed, the band between them is work in progress) and
throughput per time bucket, order queue, scheduler decision of the tick (candidates, scores, feasible
coalitions, outcomes), robots, packages, stations and charging docks (per-resource metrics), battery,
per-task cause timeline (bottleneck attribution), metrics, unsuccessful-order diagnosis (each with an
evidence span that jumps to its tick) and the event timeline.

Selecting (click or tap; works while playing): a kanban card, order, task, robot, package, decision
row, dependency-graph node, or on the map a robot, station, charging dock or shelf. The selection
panel shows its details and history (an order: its tasks, their states and events; a station:
deliveries, service and occupancy over time; a dock: sessions, energy and occupancy over time).
Filters: run (policy), order, package, task, robot, coalition (simulator task) and event type; IDs
may be typed in short form (``12`` for ``O00012``). Exports: CSV ledgers, stations/chargers CSV,
events CSV, trace JSON, metrics JSON.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Sequence, Union

from battery_tarware.orders.trace import _json_default

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Battery-TA-RWARE replay</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d2330;--muted:#6b7280;--line:#e3e6eb;--acc:#2563eb;--hi:#facc15}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.35 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);padding:8px 14px;display:flex;flex-wrap:wrap;gap:8px 10px;align-items:center}
header h1{font-size:15px;margin:0 8px 0 0}header select,header input,header button{font:inherit;padding:3px 6px;border:1px solid var(--line);border-radius:6px;background:#fff}
header button{cursor:pointer}#tick{width:min(520px,60vw)}.filters input{width:76px}.filters input.bad{border-color:#dc2626;background:#fef2f2}
main{display:grid;grid-template-columns:minmax(320px,1.05fr) minmax(320px,1fr);gap:10px;padding:10px}
@media(max-width:900px){main{grid-template-columns:1fr}}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 10px;min-width:0}
section h2{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:2px 0 6px}
h4{margin:8px 0 4px;font-size:12px}
.wide{grid-column:1/-1}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:2px 5px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:600}.scroll{max-height:260px;overflow:auto}tr.sel td{background:#e8f0ff}[data-act]{cursor:pointer}tr[data-act]:hover td{background:#f2f5fa}
#kanban{display:grid;grid-template-columns:repeat(10,minmax(86px,1fr));gap:6px;overflow-x:auto}
.col{background:#f2f4f7;border-radius:8px;padding:4px;min-height:80px}.col h3{font-size:11px;margin:2px 2px 4px;color:var(--muted);display:flex;justify-content:space-between}
.card{background:#fff;border:1px solid var(--line);border-left:4px solid #999;border-radius:6px;padding:3px 5px;margin-bottom:4px;font-size:11px;user-select:none}
.card:hover{border-color:#94a3b8}.card.of{box-shadow:0 0 0 2px var(--hi)}.card.sel{outline:2px solid var(--acc);box-shadow:none}
.card small{color:var(--muted);display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pill{display:inline-block;padding:0 6px;border-radius:9px;font-size:11px;background:#eef}.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px}
.kv b{font-weight:600}canvas{max-width:100%;display:block}.muted{color:var(--muted)}.bar{height:8px;background:#e5e7eb;border-radius:4px;overflow:hidden;width:60px;display:inline-block;vertical-align:middle}
.bar i{display:block;height:100%}.legend span{display:inline-block;margin-right:10px;font-size:11px}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:3px;vertical-align:-1px}
.selhead{display:flex;justify-content:space-between;align-items:baseline;gap:8px}.selhead b{font-size:14px}.link{color:var(--acc);cursor:pointer;text-decoration:underline}
.hint{font-size:11px;color:var(--muted);margin:2px 0 6px}
</style></head><body>
<header>
 <h1>Battery-TA-RWARE replay</h1>
 <label>run <select id="run"></select></label>
 <button id="play" title="play / pause">&#9654;</button><button id="back">&#8722;1</button><button id="fwd">+1</button>
 <input id="tick" type="range" min="0" max="0" value="0"><span id="tlabel" class="pill"></span>
 <label>speed <select id="speed"><option>1</option><option selected>4</option><option>16</option><option>64</option></select></label>
 <span class="filters">order <input id="f_order" placeholder="12 / O00012"> package <input id="f_pkg" placeholder="P00003">
 task <input id="f_task" placeholder="T00007"> robot <input id="f_robot" placeholder="agv_2"> coalition <input id="f_coal" placeholder="sim task id">
 event <input id="f_type" placeholder="type"></span><button id="clear">clear selection</button>
 <span>export <select id="export"><option value="">choose&hellip;</option><option>orders.csv</option><option>tasks.csv</option><option>packages.csv</option>
 <option>shelves.csv</option><option>robots.csv</option><option>trips.csv</option><option>stations.csv</option><option>chargers.csv</option><option>events.csv</option><option>metrics.json</option><option>trace.json</option></select></span>
</header>
<main>
 <section><h2>Warehouse</h2><div class="hint">Click a robot, a station (green), a charging dock (blue) or a shelf; click the same cell again to cycle (e.g. robot &rarr; dock).</div><canvas id="map"></canvas><div id="maplegend" class="legend"></div></section>
 <section><h2>Selected</h2><div id="selected"></div>
   <div id="selcharts"><div id="c_sel1_leg" class="legend"></div><canvas id="c_sel1" height="130"></canvas><div id="c_sel2_leg" class="legend"></div><canvas id="c_sel2" height="110"></canvas></div>
   <div id="dagbox" style="overflow-x:auto"><h4>Dependency graph of the order</h4><svg id="dag" width="100%" height="60"></svg></div><div id="seldetail" class="scroll" style="max-height:330px"></div></section>
 <section class="wide"><h2>Order tasks &mdash; kanban at this tick</h2><div id="kanban"></div></section>
 <section><h2>Order flow (cumulative)</h2><div class="hint">Gap between arrived and completed = orders in progress (or failed/cancelled). Hover for values, click to jump.</div><div id="c_flow_leg" class="legend"></div><canvas id="c_flow" height="170"></canvas></section>
 <section><h2>Throughput</h2><div class="hint" id="rate_hint"></div><div id="c_rate_leg" class="legend"></div><canvas id="c_rate" height="170"></canvas></section>
 <section><h2>Scheduler decision at this tick</h2><div id="decision" class="scroll"></div></section>
 <section><h2>Order queue (open)</h2><div id="orders" class="scroll"></div></section>
 <section><h2>Robots</h2><div id="robots" class="scroll"></div></section>
 <section><h2>Packages</h2><div id="packages" class="scroll"></div></section>
 <section><h2>Stations and charging docks</h2><div id="resources" class="scroll" style="max-height:300px"></div></section>
 <section><h2>Battery (state of charge)</h2><div id="c_soc_leg" class="legend"></div><canvas id="c_soc" height="170"></canvas></section>
 <section class="wide"><h2>Cause timeline (bottleneck attribution per task)</h2><div id="causelegend" class="legend"></div><canvas id="gantt" height="160"></canvas></section>
 <section><h2>Metrics</h2><div id="metrics" class="scroll" style="max-height:340px"></div></section>
 <section><h2>Unsuccessful orders &mdash; diagnosis</h2><div id="diag" class="scroll" style="max-height:340px"></div></section>
 <section class="wide"><h2>Event timeline</h2><div id="evhint" class="hint"></div><div id="events" class="scroll" style="max-height:300px"></div></section>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const $ = id => document.getElementById(id);
const STATES = ["pending","blocked","ready","assigned","executing","suspended","retrying","failed","completed","cancelled"];
const SCOL = {pending:"#9ca3af",blocked:"#f59e0b",ready:"#3b82f6",assigned:"#6366f1",executing:"#16a34a",suspended:"#a855f7",retrying:"#ef4444",failed:"#b91c1c",completed:"#059669",cancelled:"#6b7280"};
const CCOL = {blocked_dependency:"#f59e0b",blocked_inventory:"#fbbf24",blocked_access:"#d97706",capability:"#7c2d12",sequencing:"#3b82f6",
 coalition_unavailable:"#6366f1",charging_contention:"#06b6d4",breakdown:"#be123c",battery:"#e11d48",unreachable:"#78350f",travel:"#86efac",
 execution:"#16a34a",coalition_waiting:"#a855f7",congestion:"#f97316",failure_recovery:"#ef4444",unassigned:"#cbd5e1"};
const TYPECOL = ["#2563eb","#db2777","#059669","#d97706","#7c3aed","#0891b2","#4b5563","#65a30d"];
const ZCOL = {aisle:"#ffffff",storage:"#f3ead8",goal:"#bbf7d0",dock:"#dbeafe"};
const SELKEYS = ["task","order","robot","pkg","station","charger","shelf","coal"];
let R, ST = {sel:{}, hover:{}}, CH = {}, playing = false, timer = null;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = v => v == null ? "&ndash;" : (typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : esc(typeof v === "object" ? JSON.stringify(v) : v));
const pct = v => v == null ? "&ndash;" : (100 * v).toFixed(1) + "%";
const num = (v, d) => v == null ? "&ndash;" : Number(v).toFixed(d);
function at(hist, t){ let v = null; for (const h of hist){ if (h[0] <= t) v = h; else break; } return v; }
const ckey = c => c[0] + "," + c[1];

// ───────────────────────────────────────────── data preparation (from the trace only)
function prepare(run){
  const m = run.meta, ev = run.events, P = {run, m, ev, T0: m.first_tick, N: m.ticks};
  P.types = [...new Set(m.robots.map(r => r.type))];
  P.tasks = {}; P.orders = {}; P.decisions = {}; P.simToTask = {};
  for (const row of (run.tables?.tasks || [])) P.tasks[row.task_id] = {row, hist: []};
  for (const row of (run.tables?.orders || [])) P.orders[row.order_id] = {row, arrival: row.arrival, closed: row.closed, state: row.state};
  P.pkg = {}; P.shelf = {};
  const pk = (id, t, ch) => { const h = (P.pkg[id] = P.pkg[id] || []); const last = h.length ? h[h.length-1][1] : {}; h.push([t, Object.assign({}, last, ch)]); };
  const sh = (id, t, ch) => { const h = (P.shelf[id] = P.shelf[id] || []); const last = h.length ? h[h.length-1][1] : {}; h.push([t, Object.assign({}, last, ch)]); };
  const onShelf = {}, locCell = {};
  const idx = t => Math.max(0, Math.min(P.N - 1, t - P.T0));
  P.perArr = new Array(P.N).fill(0); P.perDone = new Array(P.N).fill(0); P.perShip = new Array(P.N).fill(0); P.perStore = new Array(P.N).fill(0);
  P.cellEv = {}; P.chargeSess = {}; const openCh = {};
  P.orderEv = {};
  for (const e of ev){
    const [t, type, robots, task, d] = e;
    if (d && d.order_id) (P.orderEv[d.order_id] = P.orderEv[d.order_id] || []).push(e);
    if (type === "order_arrived") P.perArr[idx(t)]++;
    else if (type === "order_completed") P.perDone[idx(t)]++;
    else if (type === "order_task"){ (P.tasks[d.task_id] = P.tasks[d.task_id] || {row:{task_id:d.task_id, order_id:d.order_id, op:d.op}, hist:[]}).hist.push([t, d.to, d.reason, d.attempt]); if (task != null) P.simToTask[task] = d.task_id; }
    else if (type === "schedule_decision") P.decisions[t] = d;
    else if (type === "charge_started"){ openCh[robots[0]] = {charger: d.charger, robot: robots[0], start: t, soc0: d.soc, end: null, soc1: null, reason: null};
      (P.chargeSess[d.charger] = P.chargeSess[d.charger] || []).push(openCh[robots[0]]); }
    else if (type === "charge_stopped"){ const s = openCh[robots[0]]; if (s){ s.end = t; s.soc1 = d.soc; s.reason = d.reason; delete openCh[robots[0]]; } }
    else if (type === "inventory"){
      const op = d.op;
      if (d.cell && (op === "package_shipped" || op === "package_arrived")) (P.cellEv[ckey(d.cell)] = P.cellEv[ckey(d.cell)] || []).push(e);
      if (op === "package_shipped") P.perShip[idx(t)]++;
      if (op === "package_put") P.perStore[idx(t)]++;
      if (op === "initialised"){ for (const [l, c] of Object.entries(d.locations)) locCell[l] = c;
        for (const [s, [l, pks]] of Object.entries(d.shelves)){ onShelf[s] = [...pks]; sh(s, t, {status:"stored", loc:l, cell:locCell[l]});
          for (const p of pks) pk(p, t, {status:"stored", shelf:s, loc:l, weight:d.packages[p]}); } }
      else if (op === "package_arrived") pk(d.package_id, t, {status:"arrived", shelf:null, loc:null, cell:d.cell, weight:d.weight_kg});
      else if (op === "package_reserved" || op === "slot_reserved") pk(d.package_id, t, {reserved:d.task_id});
      else if (op === "package_unreserved") pk(d.package_id, t, {reserved:null});
      else if (op === "shelf_lifted"){ sh(d.shelf_id, t, {status:"in_transit", loc:null}); for (const p of onShelf[d.shelf_id]||[]) pk(p, t, {status:"in_transit", loc:null}); }
      else if (op === "shelf_dropped"){ sh(d.shelf_id, t, {status:"dropped", cell:d.cell}); for (const p of onShelf[d.shelf_id]||[]) pk(p, t, {status:"dropped"}); }
      else if (op === "shelf_reloaded"){ sh(d.shelf_id, t, {status:"in_transit"}); for (const p of onShelf[d.shelf_id]||[]) pk(p, t, {status:"in_transit"}); }
      else if (op === "shelf_placed"){ sh(d.shelf_id, t, {status:"stored", loc:d.location_id, cell:locCell[d.location_id]}); for (const p of onShelf[d.shelf_id]||[]) pk(p, t, {status:"stored", loc:d.location_id}); }
      else if (op === "package_put"){ (onShelf[d.shelf_id] = onShelf[d.shelf_id] || []).push(d.package_id); const s = at(P.shelf[d.shelf_id]||[], t); pk(d.package_id, t, {status:s ? s[1].status : "in_transit", shelf:d.shelf_id, reserved:null}); }
      else if (op === "package_shipped"){ onShelf[d.shelf_id] = (onShelf[d.shelf_id]||[]).filter(p => p !== d.package_id); pk(d.package_id, t, {status:"shipped", shelf:null, loc:null, cell:d.cell, reserved:null}); }
      else if (op === "package_cancelled") pk(d.package_id, t, {status:"cancelled", reserved:null});
    }
  }
  P.locCell = locCell;
  P.taskIds = Object.keys(P.tasks).sort();
  const cum = a => { let s = 0; return a.map(v => s += v); };
  P.cumArr = cum(P.perArr); P.cumDone = cum(P.perDone);
  P.B = P.N >= 1000 ? 100 : Math.max(10, Math.round(P.N / 15));
  P.stations = ((m.zones || {}).goal || []).map(c => ({key: ckey(c), cell: c, name: "G" + c[0] + "_" + c[1]}));
  P.chargers = (m.chargers || []).map(c => ({...c, key: ckey(c.cell)}));
  P.res = run.metrics?.resources || null;
  P.A = Object.fromEntries((m.activities || []).map((a, k) => [a, k]));
  return P;
}
function bucket(per, B){ const out = []; for (let i = 0; i < per.length; i += B){ let s = 0; for (let j = i; j < Math.min(per.length, i + B); j++) s += per[j]; out.push(s); } return out; }

// ───────────────────────────────────────────── selection and filters
function tickNow(){ return R.T0 + Number($("tick").value); }
function frame(t){ return R.run.frames[Math.max(0, Math.min(R.N-1, t - R.T0))]; }
function taskState(id, t){ const h = at(R.tasks[id].hist, t); return h ? h : null; }
function resolveId(kind, text){
  const s = (text || "").trim(); if (!s) return "";
  const pool = kind === "order" ? Object.keys(R.orders) : kind === "task" ? R.taskIds : kind === "pkg" ? Object.keys(R.pkg) : kind === "robot" ? R.m.robots.map(r => r.name) : null;
  if (!pool) return s;
  const hit = pool.find(id => id.toLowerCase() === s.toLowerCase()); if (hit) return hit;
  if (/^\d+$/.test(s)){ const n = Number(s), c = pool.find(id => /\d+$/.test(id) && Number(id.match(/\d+$/)[0]) === n && kind !== "robot"); if (c) return c; }
  const c = pool.filter(id => id.toLowerCase().includes(s.toLowerCase())); return c.length === 1 ? c[0] : null;
}
function filters(){ const F = {};
  for (const [k, kind] of [["order","order"],["pkg","pkg"],["task","task"],["robot","robot"]]){ const inp = $("f_" + k), v = resolveId(kind, inp.value); inp.classList.toggle("bad", v === null); F[k] = v || ""; }
  F.coal = $("f_coal").value.trim(); F.type = $("f_type").value.trim(); return F; }
function taskMatches(id, F){ const r = R.tasks[id].row;
  if (F.order && r.order_id !== F.order) return false; if (F.task && id !== F.task) return false;
  if (F.pkg && r.package_id !== F.pkg) return false; if (F.coal && !(r.sim_tasks||"").split("|").includes(F.coal)) return false; return true; }
function selOrder(){ const s = ST.sel; return s.order || (s.task && R.tasks[s.task] ? R.tasks[s.task].row.order_id : null); }
function robotOrderAt(fr){ const tid = fr[4] >= 0 ? R.simToTask[fr[4]] : null; return tid ? [tid, R.tasks[tid].row.order_id] : [null, null]; }
function select(s){ ST.sel = s; const h = ["run=" + $("run").value, "t=" + tickNow()]; for (const k of SELKEYS) if (s[k] != null) h.push(k + "=" + encodeURIComponent(s[k]));
  try { history.replaceState(null, "", "#" + h.join("&")); } catch (e) {} redraw(); }
// one delegated handler for every clickable element (data-act), on press: survives re-rendering during playback
document.querySelector("main").addEventListener("pointerdown", ev => {
  if (ev.button !== 0) return; const el = ev.target.closest("[data-act]"); if (!el) return;
  const d = el.dataset, s = {};
  for (const k of SELKEYS) if (d[k] != null && d[k] !== "") s[k] = d[k];
  if (s.coal != null && R.simToTask[s.coal]){ s.task = R.simToTask[s.coal]; delete s.coal; }
  if (d.t) $("tick").value = Math.max(0, Math.min(R.N - 1, Number(d.t) - R.T0));
  if (Object.keys(s).length) select(s); else redraw();
});

// ───────────────────────────────────────────── charts (axes, legend, hover read-out, click to jump)
function niceMax(v){ if (!(v > 0)) return 1; const p = Math.pow(10, Math.floor(Math.log10(v))); for (const k of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (k * p >= v) return k * p; return 10 * p; }
function niceInt(v){ const p = Math.pow(10, Math.floor(Math.log10(v))); for (const k of [1, 2, 4, 8, 10]) if (k * p >= v) return k * p; return 10 * p; }
function chart(id, spec){
  CH[id] = spec; const c = $(id); if (!c) return; const w = Math.max(220, c.parentElement.clientWidth - 20); c.width = w;
  const h = c.height, g = c.getContext("2d"), L = 44, Rm = 10, Tm = 8, Bm = 18, pw = w - L - Rm, ph = h - Tm - Bm;
  g.clearRect(0, 0, w, h); g.font = "10px system-ui,sans-serif";
  const T0 = R.T0, N = R.N, X = tt => L + (tt - T0) / Math.max(1, N - 1) * pw;
  let mx = spec.ymax; if (mx == null){ mx = 0; for (const s of spec.series) for (const v of s.data) if (v != null && v > mx) mx = v; mx = spec.int ? (mx <= 20 ? Math.max(4, Math.ceil(mx / 4) * 4) : niceInt(mx)) : niceMax(mx); }
  const Y = v => Tm + ph - Math.max(0, Math.min(1, v / mx)) * ph, yf = spec.yfmt || (spec.int ? (v => String(Math.round(v))) : (v => (Math.round(v * 100) / 100).toString()));
  g.strokeStyle = "#eef0f3"; g.fillStyle = "#6b7280"; g.textAlign = "right";
  for (let k = 0; k <= 4; k++){ const v = mx * k / 4, y = Y(v); g.beginPath(); g.moveTo(L, y); g.lineTo(L + pw, y); g.stroke(); g.fillText(yf(v), L - 4, y + 3); }
  g.textAlign = "center"; for (let k = 0; k <= 5; k++){ const tt = Math.round(T0 + k / 5 * (N - 1)); g.fillText(String(tt), Math.min(L + pw - 10, Math.max(L + 10, X(tt))), h - 4); }
  g.textAlign = "left";
  const bars = spec.series.filter(s => s.kind === "bar"), groups = [...new Set(bars.map(s => s.group ?? s.name))];
  const stackBase = {};
  for (const s of spec.series){
    if (s.kind === "bar"){ const B = s.bucket, gi = groups.indexOf(s.group ?? s.name), ng = groups.length;
      g.fillStyle = s.color; g.globalAlpha = s.alpha ?? 0.85;
      s.data.forEach((v, i) => { if (!v) return; const x0 = X(T0 + i * B), x1 = X(Math.min(T0 + N - 1, T0 + (i + 1) * B)), bw = Math.max(1, (x1 - x0 - 2) / ng);
        const key = gi + ":" + i, base = stackBase[key] || 0; stackBase[key] = base + v;
        g.fillRect(x0 + 1 + gi * bw, Y(base + v), bw, Y(base) - Y(base + v)); });
      g.globalAlpha = 1; continue; }
    const step = Math.max(1, Math.floor(N / pw / 2)), pts = [];
    for (let i = 0; i < s.data.length; i += step){ const v = s.data[i]; if (v != null) pts.push([X(T0 + i), Y(v)]); }
    const lastI = s.data.length - 1; if (lastI >= 0 && s.data[lastI] != null) pts.push([X(T0 + lastI), Y(s.data[lastI])]);
    if (!pts.length) continue;
    if (s.fill){ g.fillStyle = s.fill; g.beginPath(); g.moveTo(pts[0][0], Y(0)); for (const p of pts) g.lineTo(p[0], p[1]); g.lineTo(pts[pts.length-1][0], Y(0)); g.closePath(); g.fill(); }
    g.strokeStyle = s.color; g.lineWidth = s.width || 1.5; g.globalAlpha = s.alpha ?? 1; g.beginPath();
    pts.forEach((p, k) => { if (!k) g.moveTo(p[0], p[1]); else if (s.kind === "step"){ g.lineTo(p[0], pts[k-1][1]); g.lineTo(p[0], p[1]); } else g.lineTo(p[0], p[1]); });
    g.stroke(); g.globalAlpha = 1; g.lineWidth = 1;
  }
  const cx = X(tickNow()); g.strokeStyle = "#111827"; g.beginPath(); g.moveTo(cx, Tm); g.lineTo(cx, Tm + ph); g.stroke();
  const hv = ST.hover[id];
  if (hv != null){ const hx = X(hv); g.strokeStyle = "#9ca3af"; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(hx, Tm); g.lineTo(hx, Tm + ph); g.stroke(); g.setLineDash([]);
    const lines = ["tick " + hv]; for (const s of spec.series){ if (s.noTip) continue; const i = s.kind === "bar" ? Math.floor((hv - T0) / s.bucket) : hv - T0, v = s.data[i];
      lines.push(s.name + ": " + (v == null ? "–" : yf(v)) + (s.kind === "bar" ? " (ticks " + (T0 + Math.floor((hv - T0) / s.bucket) * s.bucket) + "–" + (T0 + (Math.floor((hv - T0) / s.bucket) + 1) * s.bucket - 1) + ")" : "")); }
    const tw = Math.max(...lines.map(l => g.measureText(l).width)) + 10, th = lines.length * 13 + 6, tx = hx + 8 + tw > w ? hx - 8 - tw : hx + 8;
    g.fillStyle = "rgba(255,255,255,.95)"; g.strokeStyle = "#d1d5db"; g.fillRect(tx, Tm + 2, tw, th); g.strokeRect(tx, Tm + 2, tw, th);
    g.fillStyle = "#111827"; lines.forEach((l, k) => g.fillText(l, tx + 5, Tm + 15 + k * 13)); }
  const leg = $(id + "_leg"); if (leg) leg.innerHTML = spec.series.filter(s => !s.noLegend).map(s => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("") + (spec.legendExtra || "");
  c._geo = {L, pw};
}
function hookChart(id){ const c = $(id);
  const tickAt = ev => { const r = c.getBoundingClientRect(), x = (ev.clientX - r.left) * (c.width / r.width), g = c._geo; if (!g || x < g.L) return null;
    return Math.max(R.T0, Math.min(R.T0 + R.N - 1, Math.round(R.T0 + (x - g.L) / g.pw * (R.N - 1)))); };
  c.addEventListener("mousemove", ev => { ST.hover[id] = tickAt(ev); if (CH[id]) chart(id, CH[id]); });
  c.addEventListener("mouseleave", () => { delete ST.hover[id]; if (CH[id]) chart(id, CH[id]); });
  c.addEventListener("pointerdown", ev => { const tt = tickAt(ev); if (tt != null) setTick(tt); }); }

function drawCharts(t){
  const B = R.B;
  chart("c_flow", {series:[
    {name:"orders arrived (cumulative)", color:"#2563eb", data:R.cumArr, fill:"rgba(37,99,235,.10)", width:2},
    {name:"orders completed (cumulative)", color:"#059669", data:R.cumDone, fill:"rgba(5,150,105,.18)", width:2},
    {name:"open orders", color:"#f59e0b", data:(R.run.series?.open_orders || []), kind:"step", width:1}], int:true});
  $("rate_hint").textContent = "Per " + B + "-tick bucket. Bars: orders completed and packages delivered (shipped + stored); line: orders arrived (demand).";
  chart("c_rate", {series:[
    {name:"orders completed", color:"#059669", data:bucket(R.perDone, B), kind:"bar", bucket:B, group:"o"},
    {name:"packages shipped", color:"#db2777", data:bucket(R.perShip, B), kind:"bar", bucket:B, group:"d"},
    {name:"packages stored (inbound)", color:"#f472b6", data:bucket(R.perStore, B), kind:"bar", bucket:B, group:"d"},
    {name:"orders arrived", color:"#2563eb", data:stepBuckets(bucket(R.perArr, B), B), kind:"step", width:2}], int:true});
  const fr = R.run.frames, sel = ST.sel.robot, ser = [];
  R.m.robots.forEach((rb, i) => { if (rb.name === sel) return; ser.push({name:rb.name, color:TYPECOL[R.types.indexOf(rb.type) % 8], data:fr.map(f => f[i][2]), alpha:sel ? .18 : .45, width:1, noLegend:true, noTip:true}); });
  ser.push({name:"fleet mean", color:"#111827", data:fr.map(f => f.reduce((a, r) => a + r[2], 0) / f.length), width:2});
  if (sel){ const i = R.m.robots.findIndex(r => r.name === sel); if (i >= 0) ser.push({name:sel + " (selected)", color:"#dc2626", data:fr.map(f => f[i][2]), width:2.5}); }
  chart("c_soc", {series:ser, ymax:1, yfmt:v => Math.round(v * 100) + "%", legendExtra:R.types.map((ty, k) => `<span><i style="background:${TYPECOL[k%8]};opacity:.5"></i>${esc(ty)}</span>`).join("")});
}
function stepBuckets(b, B){ const out = []; b.forEach(v => { for (let j = 0; j < B; j++) out.push(v); }); return out.slice(0, R.N); }

// ───────────────────────────────────────────── panels
function drawMap(t){
  const c = $("map"), m = R.m, [H, W] = m.shape, cs = Math.max(10, Math.floor(Math.min(620, c.parentElement.clientWidth - 20) / W));
  c.width = W*cs; c.height = H*cs; const g = c.getContext("2d");
  g.fillStyle = "#374151"; g.fillRect(0,0,c.width,c.height);
  for (const [z, cells] of Object.entries(m.zones)) for (const [y,x] of cells){ g.fillStyle = ZCOL[z] || "#e5e7eb"; g.fillRect(x*cs, y*cs, cs-1, cs-1); }
  const fr = frame(t), occ = new Set(fr.map(r => r[0] + "," + r[1]));
  g.font = Math.max(8, Math.floor(cs * 0.36)) + "px system-ui,sans-serif"; g.textAlign = "center";
  for (const ch of R.chargers){ const [y, x] = ch.cell, busy = fr.some(r => r[0] === y && r[1] === x && r[6] === R.A.charging);
    g.fillStyle = busy ? "#3b82f6" : "#93c5fd"; g.fillRect(x*cs+2, y*cs+2, cs-5, cs-5); if (cs >= 14 && !occ.has(ch.key)){ g.fillStyle = "#1e3a8a"; g.fillText("C" + ch.id, x*cs + cs/2, y*cs + cs*0.62); } }
  for (const st of R.stations){ const [y, x] = st.cell; g.strokeStyle = "#16a34a"; g.lineWidth = 2; g.strokeRect(x*cs+2, y*cs+2, cs-5, cs-5); g.lineWidth = 1;
    if (cs >= 14 && !occ.has(st.key)){ g.fillStyle = "#166534"; g.fillText("G", x*cs + cs/2, y*cs + cs*0.62); } }
  g.textAlign = "left";
  const shelfFull = {}; for (const ph of Object.values(R.pkg)){ const q = at(ph, t); if (q && q[1].shelf && q[1].status === "stored") shelfFull[q[1].shelf] = true; }
  for (const [s, h] of Object.entries(R.shelf)){ const v = at(h, t); if (v && v[1].status === "stored" && v[1].cell){ const [y,x] = v[1].cell;
      g.fillStyle = shelfFull[s] ? "#a16207" : "#d6b98c"; g.fillRect(x*cs+2, y*cs+2, cs-5, cs-5);
      if (ST.sel.shelf === s){ g.strokeStyle = "#111827"; g.lineWidth = 2; g.strokeRect(x*cs+1, y*cs+1, cs-3, cs-3); g.lineWidth = 1; } } }
  const so = selOrder(), selT = ST.sel.task, selSim = ST.sel.coal != null ? Number(ST.sel.coal) : null;
  for (const [s, y, x, st] of (R.run.shelves[t - R.T0] || [])){ g.fillStyle = st === "dropped" ? "#f97316" : "#a16207"; g.fillRect(x*cs+1, y*cs+1, cs-3, cs-3); }
  fr.forEach((r, i) => { const [y, x, soc, status] = r, rb = R.m.robots[i], [tid, oid] = robotOrderAt(r);
    g.beginPath(); g.arc(x*cs+cs/2, y*cs+cs/2, cs*0.34, 0, 7); g.fillStyle = status >= 3 ? "#111827" : TYPECOL[R.types.indexOf(rb.type) % 8]; g.fill();
    const hi = ST.sel.robot === rb.name || (selT && tid === selT) || (!selT && so && oid === so) || (selSim != null && r[4] === selSim);
    if (hi){ g.lineWidth = 3; g.strokeStyle = "#facc15"; g.stroke(); g.lineWidth = 1; }
    g.fillStyle = soc < 0.3 ? "#dc2626" : "#16a34a"; g.fillRect(x*cs+2, y*cs+cs-4, Math.max(1,(cs-5)*soc), 3); });
  const selCell = ST.sel.station || (ST.sel.charger != null ? (R.chargers.find(c => String(c.id) === String(ST.sel.charger)) || {}).key : null);
  if (selCell){ const [y, x] = selCell.split(",").map(Number); g.strokeStyle = "#facc15"; g.lineWidth = 3; g.strokeRect(x*cs, y*cs, cs-1, cs-1); g.lineWidth = 1; }
  c._cs = cs;
  $("maplegend").innerHTML = R.types.map((ty, k) => `<span><i style="background:${TYPECOL[k%8]}"></i>${esc(ty)}</span>`).join("") +
    '<span><i style="background:#a16207"></i>shelf with package</span><span><i style="background:#d6b98c"></i>empty shelf</span><span><i style="background:#93c5fd"></i>dock (C)</span><span><i style="background:#3b82f6"></i>dock in use</span><span><i style="background:#bbf7d0;border:1px solid #16a34a"></i>station (G)</span><span><i style="background:#111827"></i>broken/stranded</span><span><i style="background:#facc15"></i>selected</span>';
}

function drawKanban(t){
  const F = filters(), cols = {}, so = selOrder(); STATES.forEach(s => cols[s] = []);
  for (const id of R.taskIds){ if (!taskMatches(id, F)) continue; const h = taskState(id, t); if (!h) continue; cols[h[1]].push([id, h]); }
  const tot = {}; STATES.forEach(s => tot[s] = cols[s].length);
  const keep = ([id]) => id === ST.sel.task || (so && R.tasks[id].row.order_id === so);   // the selection stays visible
  for (const s of STATES) cols[s].sort((a, b) => keep(b) - keep(a));
  cols.completed = [...cols.completed.filter(keep), ...cols.completed.filter(x => !keep(x)).sort((a,b) => b[1][0]-a[1][0])].slice(0, Math.max(12, cols.completed.filter(keep).length));
  cols.cancelled = [...cols.cancelled.filter(keep), ...cols.cancelled.filter(x => !keep(x)).slice(-12)];
  $("kanban").innerHTML = STATES.map(s => `<div class="col"><h3><span>${s}</span><span>${tot[s]}</span></h3>` + cols[s].slice(0, 40).map(([id, h]) => {
      const r = R.tasks[id].row, cls = ST.sel.task === id ? " sel" : (so && r.order_id === so ? " of" : "");
      return `<div class="card${cls}" style="border-left-color:${SCOL[s]}" data-act data-task="${id}"><b>${id}</b> ${esc(r.op||"")}<small>${esc(r.order_id)} ${esc(r.package_id||r.shelf_id||"")}${h[3]>1?" &middot; try "+h[3]:""} &middot; ${t - h[0]}t</small>${h[2]?`<small title="${esc(h[2])}">${esc(h[2])}</small>`:""}</div>`; }).join("") + "</div>").join("");
}

function drawOrders(t){
  const F = filters(), rows = [];
  for (const [id, o] of Object.entries(R.orders)){ if (o.arrival > t || (o.closed != null && o.closed <= t)) continue; if (F.order && id !== F.order) continue;
    const tids = (o.row.tasks||"").split("|").filter(Boolean), done = tids.filter(x => R.tasks[x] && (taskState(x, t)||[])[1] === "completed").length;
    rows.push(`<tr class="${selOrder()===id?"sel":""}" data-act data-order="${id}"><td>${id}</td><td>${esc(o.row.kind)}</td><td>${t-o.arrival}</td><td>${fmt(o.row.priority)}</td><td>${o.row.due==null?"&ndash;":o.row.due-t}</td><td>${done}/${tids.length}</td></tr>`); }
  $("orders").innerHTML = `<table><tr><th>order</th><th>kind</th><th>age</th><th>prio</th><th>due in</th><th>tasks done</th></tr>${rows.join("")}</table>` + (rows.length ? "" : '<p class="muted">no open orders</p>');
}

function drawDecision(t){
  const d = R.decisions[t]; if (!d){ $("decision").innerHTML = '<p class="muted">no decision record at this tick (no eligible task, or a coordinator without a decision log)</p>'; return; }
  const rows = d.tasks.map(r => `<tr data-act data-coal="${r[0]}"><td>${r[0]}</td><td>${esc(r[1]||"")}</td><td>${fmt(r[2])}</td><td>${r[3]}</td><td>${fmt(r[4])}</td><td>${esc(r[5])}</td><td>${esc((d.assigned[String(r[0])]||[]).map(i => R.m.robots[i].name).join(", "))}</td></tr>`);
  $("decision").innerHTML = `<div class="kv"><b>rule</b><span>${esc(d.rule)}</span><b>free robots</b><span>${d.free}</span><b>candidate pairs</b><span>${d.pairs}</span></div>
   <table><tr><th>sim task</th><th>task</th><th>score (sorts first)</th><th>coalitions</th><th>best est.</th><th>outcome</th><th>crew</th></tr>${rows.join("")}</table>`;
}

function drawRobots(t){
  const fr = frame(t), F = filters(), m = R.m;
  const rows = fr.map((r, i) => { const rb = m.robots[i]; if (F.robot && rb.name !== F.robot) return "";
    const task = r[4] >= 0 ? (R.simToTask[r[4]] || "sim "+r[4]) : "";
    return `<tr class="${ST.sel.robot===rb.name?"sel":""}" data-act data-robot="${rb.name}"><td>${esc(rb.name)}</td><td>${esc(m.activities[r[6]])}</td><td><span class="bar"><i style="width:${Math.round(100*r[2])}%;background:${r[2]<0.3?"#dc2626":"#16a34a"}"></i></span> ${Math.round(100*r[2])}%</td><td>${esc(task)}</td><td>${r[5]>=0?esc(m.roles[r[5]]):""}</td><td>${r[0]},${r[1]}</td></tr>`; });
  $("robots").innerHTML = `<table><tr><th>robot</th><th>activity</th><th>battery</th><th>task</th><th>role</th><th>cell</th></tr>${rows.join("")}</table>`;
}

function drawPackages(t){
  const F = filters(), rows = [];
  for (const id of Object.keys(R.pkg).sort()){ if (F.pkg && id !== F.pkg) continue; const v = at(R.pkg[id], t); if (!v) continue; const p = v[1];
    if (!F.pkg && ST.sel.pkg !== id && (p.status === "shipped" || p.status === "cancelled") && v[0] < t - 50) continue;
    rows.push(`<tr class="${ST.sel.pkg===id?"sel":""}" data-act data-pkg="${id}"><td>${id}</td><td>${esc(p.status)}</td><td>${esc(p.shelf||"")}</td><td>${esc(p.loc||"")}</td><td>${esc(p.reserved||"")}</td><td>${fmt(p.weight)}</td></tr>`); }
  $("packages").innerHTML = `<table><tr><th>package</th><th>status</th><th>shelf</th><th>location</th><th>reserved by</th><th>kg</th></tr>${rows.join("")}</table>`;
}

function resRow(kind, key){ const r = R.res; if (!r) return null; return kind === "station" ? r.stations.find(s => ckey(s.cell) === key) : r.chargers.find(c => String(c.charger) === String(key)); }
function drawResources(t){
  const fr = frame(t), on = key => fr.map((r, i) => r[0] + "," + r[1] === key ? R.m.robots[i].name : null).filter(Boolean);
  let s = `<table><tr><th>station</th><th>cell</th><th>shipped</th><th>received</th><th>per 1000 t</th><th>stage visits</th><th>service t</th><th>occupancy</th><th>robots on it now</th></tr>`;
  for (const st of R.stations){ const x = resRow("station", st.key) || {};
    s += `<tr class="${ST.sel.station===st.key?"sel":""}" data-act data-station="${st.key}"><td>${st.name}</td><td>${st.key}</td><td>${fmt(x.shipped)}</td><td>${fmt(x.received)}</td><td>${num(x.deliveries_per_1000, 1)}</td><td>${fmt(x.stage_visits)}</td><td>${num(x.service_ticks_mean, 1)}</td><td>${pct(x.occupancy)}</td><td>${esc(on(st.key).join(", "))}</td></tr>`; }
  s += `</table><table style="margin-top:8px"><tr><th>dock</th><th>cell</th><th>sessions</th><th>robots</th><th>occupancy</th><th>energy Wh</th><th>session t</th><th>SoC in&rarr;out</th><th>now</th></tr>`;
  for (const ch of R.chargers){ const x = resRow("charger", ch.id) || {}, now = fr.map((r, i) => r[0] === ch.cell[0] && r[1] === ch.cell[1] && r[6] === R.A.charging ? R.m.robots[i].name + " " + Math.round(100 * r[2]) + "%" : null).filter(Boolean);
    s += `<tr class="${String(ST.sel.charger)===String(ch.id)?"sel":""}" data-act data-charger="${ch.id}"><td>C${ch.id}</td><td>${ch.key}</td><td>${fmt(x.sessions)}</td><td>${fmt(x.robots)}</td><td>${pct(x.occupancy)}</td><td>${num(x.energy_wh, 0)}</td><td>${num(x.session_ticks_mean, 1)}</td><td>${x.soc_start_mean == null ? "&ndash;" : pct(x.soc_start_mean) + " &rarr; " + pct(x.soc_end_mean)}</td><td>${esc(now.join(", ") || "free")}</td></tr>`; }
  const q = R.res?.dock_queue, nq = fr.filter(r => r[6] === R.A.dock_queue).length;
  s += `</table><p class="muted">Whole-run figures from <code>metrics.resources</code>; "now" columns at this tick. Dock queue (fleet-wide): ${nq} robot(s) now${q ? `, mean ${num(q.mean, 2)}, max ${q.max}` : ""}.</p>`;
  $("resources").innerHTML = s;
}

function drawGantt(t){
  const c = $("gantt"), w = c.parentElement.clientWidth - 20, F = filters(); c.width = w;
  let ids = R.taskIds.filter(id => taskMatches(id, F) && R.run.spans[id]); const so = selOrder();
  if (so) ids = ids.filter(id => R.tasks[id].row.order_id === so);
  else if (!F.order && !F.task && !F.pkg && !F.coal) ids = ids.filter(id => R.run.spans[id].some(s => s[0] <= t && s[1] >= t - 30)).slice(0, 40);
  const rh = 12, h = Math.max(40, ids.length * rh + 14); c.height = h; const g = c.getContext("2d"); g.clearRect(0,0,w,h);
  const lab = 64, sc = (w - lab) / R.N; c._lab = lab; c._sc = sc;
  ids.forEach((id, k) => { g.fillStyle = id === ST.sel.task ? "#2563eb" : "#374151"; g.font = (id === ST.sel.task ? "bold " : "") + "10px sans-serif"; g.fillText(id, 2, k*rh + 10);
    for (const [a, b, cause] of R.run.spans[id]){ g.fillStyle = CCOL[cause] || "#999"; g.fillRect(lab + (a - R.T0) * sc, k*rh + 2, Math.max(1, (b - a + 1) * sc), rh - 3); } });
  const cx = lab + (t - R.T0) * sc; g.strokeStyle = "#111"; g.beginPath(); g.moveTo(cx, 0); g.lineTo(cx, h); g.stroke();
  $("causelegend").innerHTML = Object.entries(CCOL).map(([k, v]) => `<span><i style="background:${v}"></i>${k}</span>`).join("") + (so ? ` <b>showing order ${so}</b>` : "");
}

function hist(rows, head){ return `<table><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr>${rows.join("")}</table>`; }
function selHead(title, sub){ return `<div class="selhead"><b>${title}</b><span class="muted">${sub || ""}</span></div>`; }
function drawSelected(t){
  const s = ST.sel, out = []; let charts = null, dag = false;
  if (s.task && R.tasks[s.task]){ const T = R.tasks[s.task], r = T.row, now = taskState(s.task, t), fr = frame(t);
    const crew = fr.map((x, i) => R.simToTask[x[4]] === s.task ? R.m.robots[i].name : null).filter(Boolean);
    out.push(selHead(`Task ${s.task} &middot; ${esc(r.op)}`, now ? `now <span class="pill" style="background:${SCOL[now[1]]};color:#fff">${now[1]}</span>` : "not yet created"));
    out.push(`<div class="kv"><b>order</b><span class="link" data-act data-order="${esc(r.order_id)}">${esc(r.order_id)}</span><b>package / shelf</b><span>${r.package_id ? `<span class="link" data-act data-pkg="${esc(r.package_id)}">${esc(r.package_id)}</span>` : ""} / ${esc(r.shelf_id||"")}</span><b>depends on</b><span>${(r.depends_on||"").split("|").filter(Boolean).map(x => `<span class="link" data-act data-task="${x}">${x}</span>`).join(" ")}</span><b>attempts</b><span>${fmt(r.attempt)} (sim ${esc(r.sim_tasks||"")})</span><b>crew now</b><span>${crew.map(n => `<span class="link" data-act data-robot="${n}">${n}</span>`).join(", ") || "&ndash;"}</span><b>last failure</b><span>${esc(r.failure_reason||"")} ${r.blocking_resource?"&middot; "+esc(r.blocking_resource):""}</span></div>`);
    out.push("<h4>State history (click a row to jump)</h4>" + hist(T.hist.map(h => `<tr class="${now && h[0] === now[0] && h[1] === now[1] ? "sel" : ""}" data-act data-t="${h[0]}"><td>${h[0]}</td><td>${h[1]}</td><td>${esc(h[2]||"")}</td></tr>`), ["tick", "state", "reason"]));
    dag = true;
  } else if (s.order && R.orders[s.order]){ const o = R.orders[s.order], r = o.row, tids = (r.tasks||"").split("|").filter(Boolean);
    const open = o.arrival <= t && (o.closed == null || o.closed > t), st = o.arrival > t ? "not arrived yet" : open ? "open" : r.state;
    out.push(selHead(`Order ${s.order} &middot; ${esc(r.kind)}`, `at this tick: <b>${st}</b>`));
    out.push(`<div class="kv"><b>arrival</b><span class="link" data-act data-t="${o.arrival}">${o.arrival}</span><b>due</b><span>${fmt(r.due)}</span><b>priority</b><span>${fmt(r.priority)}</span><b>final state</b><span>${esc(r.state)}${r.closed != null ? ` at <span class="link" data-act data-t="${r.closed}">${r.closed}</span>` : ""}</span><b>lead time</b><span>${fmt(r.lead_time)}</span><b>reason</b><span>${esc(r.reason||"")}</span></div>`);
    out.push("<h4>Tasks (click to select)</h4>" + hist(tids.map(id => { const T = R.tasks[id]; if (!T) return ""; const h = taskState(id, t); return `<tr data-act data-task="${id}"><td>${id}</td><td>${esc(T.row.op)}</td><td>${h ? `<span class="pill" style="background:${SCOL[h[1]]};color:#fff">${h[1]}</span>` : "&ndash;"}</td><td>${esc(T.row.package_id || T.row.shelf_id || "")}</td><td>${esc((T.row.depends_on||"").replace(/\|/g, " "))}</td><td>${fmt(T.row.attempt)}</td></tr>`; }), ["task", "op", "state now", "package / shelf", "after", "attempts"]));
    out.push("<h4>Order events (click to jump)</h4>" + hist((R.orderEv[s.order] || []).slice(0, 80).map(e => `<tr class="${e[0] === t ? "sel" : ""}" data-act data-t="${e[0]}"><td>${e[0]}</td><td>${esc(e[1])}${e[4].op && e[1] !== "order_task" ? " &middot; " + esc(e[4].op) : ""}</td><td>${esc(e[4].task_id || "")}</td><td>${esc(e[1] === "order_task" ? (e[4].frm || "") + " → " + e[4].to + (e[4].reason ? " (" + e[4].reason + ")" : "") : (e[4].reason || ""))}</td></tr>`), ["tick", "event", "task", "detail"]));
    dag = true;
  } else if (s.robot && R.m.robots.some(r => r.name === s.robot)){ const i = R.m.robots.findIndex(r => r.name === s.robot), rb = R.m.robots[i], f = frame(t)[i];
    out.push(selHead(`Robot ${esc(rb.name)} &middot; ${esc(rb.type)}`, `now: ${esc(R.m.activities[f[6]])}, ${Math.round(100 * f[2])}%`));
    out.push(`<div class="kv"><b>capabilities</b><span>${esc(rb.capabilities.join(", "))}</span><b>payload</b><span>${rb.payload_kg} kg</span><b>battery</b><span>${rb.capacity_wh} Wh</span><b>task now</b><span>${f[4] >= 0 && R.simToTask[f[4]] ? `<span class="link" data-act data-task="${R.simToTask[f[4]]}">${R.simToTask[f[4]]}</span>` : (f[4] >= 0 ? "sim " + f[4] : "&ndash;")}</span></div>`);
    const pr = (R.run.metrics?.robot?.per_robot || [])[i]; if (pr) out.push("<h4>Whole run</h4><div class='kv'>" + Object.entries(pr).map(([k, v]) => `<b>${k}</b><span>${fmt(v)}</span>`).join("") + "</div>");
    const fr = R.run.frames, B = R.B, busy = [], chg = [];
    for (let k = 0; k < R.N; k += B){ let b = 0, c = 0, n = 0; for (let j = k; j < Math.min(R.N, k + B); j++, n++){ const a = fr[j][i][6]; if (a === R.A.charging || a === R.A.to_charger || a === R.A.dock_queue) c++; else if (a !== R.A.idle && a !== R.A.failed && a !== R.A.stranded) b++; } busy.push(b / n); chg.push(c / n); }
    charts = [{series:[{name:"state of charge", color:"#dc2626", data:fr.map(x => x[i][2]), width:2}], ymax:1, yfmt:v => Math.round(v * 100) + "%"},
              {series:[{name:"busy share", color:"#16a34a", data:busy, kind:"bar", bucket:B, group:"a"}, {name:"charging share (incl. travel to dock)", color:"#3b82f6", data:chg, kind:"bar", bucket:B, group:"a"}], ymax:1, yfmt:v => Math.round(v * 100) + "%"}];
  } else if (s.pkg && R.pkg[s.pkg]){
    out.push(selHead(`Package ${s.pkg}`, ""));
    out.push("<h4>History (click to jump)</h4>" + hist(R.pkg[s.pkg].map(([tt, p]) => `<tr data-act data-t="${tt}"><td>${tt}</td><td>${esc(p.status)}</td><td>${esc(p.shelf||"")}</td><td>${esc(p.loc||"")}</td><td>${p.reserved ? `<span class="link" data-act data-task="${esc(p.reserved)}">${esc(p.reserved)}</span>` : ""}</td></tr>`), ["tick", "status", "shelf", "location", "reserved by"]));
  } else if (s.station && R.stations.some(x => x.key === s.station)){ const st = R.stations.find(x => x.key === s.station), x = resRow("station", st.key) || {};
    const fr = frame(t), here = fr.map((r, i) => r[0] + "," + r[1] === st.key ? R.m.robots[i].name + " (" + R.m.activities[r[6]] + ")" : null).filter(Boolean);
    out.push(selHead(`Station ${st.name}`, `cell ${st.key}`));
    out.push(`<div class="kv"><b>packages shipped (outbound)</b><span>${fmt(x.shipped)}</span><b>packages received (inbound)</b><span>${fmt(x.received)}</span><b>deliveries / 1000 ticks</b><span>${num(x.deliveries_per_1000, 1)}</span><b>task stages completed here</b><span>${fmt(x.stage_visits)}</span><b>mean service time</b><span>${num(x.service_ticks_mean, 2)} ticks</span><b>occupancy</b><span>${pct(x.occupancy)} (${fmt(x.occupied_ticks)} ticks)</span><b>robot-ticks working / waiting for partner / blocked</b><span>${fmt(x.working_ticks)} / ${fmt(x.waiting_partner_ticks)} / ${fmt(x.blocked_ticks)}</span><b>distinct robots</b><span>${fmt(x.robots)}</span><b>on the cell now</b><span>${esc(here.join(", ") || "nobody")}</span></div>`);
    const evs = R.cellEv[st.key] || [], ship = new Array(R.N).fill(0), recv = new Array(R.N).fill(0);
    for (const e of evs){ const k = Math.max(0, Math.min(R.N - 1, e[0] - R.T0)); (e[4].op === "package_shipped" ? ship : recv)[k]++; }
    const B = R.B, occ = [], work = [], frs = R.run.frames, [cy, cx] = st.cell;
    for (let k = 0; k < R.N; k += B){ let o = 0, w = 0, n = 0; for (let j = k; j < Math.min(R.N, k + B); j++, n++){ let here2 = false, wk = false; for (const r of frs[j]) if (r[0] === cy && r[1] === cx){ here2 = true; if (r[6] === R.A.working) wk = true; } if (here2) o++; if (wk) w++; } occ.push((o - w) / n); work.push(w / n); }
    charts = [{series:[{name:"shipped here", color:"#db2777", data:bucket(ship, B), kind:"bar", bucket:B, group:"s"}, {name:"received here", color:"#2563eb", data:bucket(recv, B), kind:"bar", bucket:B, group:"r"}], int:true},
              {series:[{name:"working on the cell", color:"#16a34a", data:work, kind:"bar", bucket:B, group:"o"}, {name:"occupied, not working", color:"#f59e0b", data:occ, kind:"bar", bucket:B, group:"o"}], ymax:1, yfmt:v => Math.round(v * 100) + "%"}];
    out.push("<h4>Deliveries at this station (click to jump)</h4>" + hist(evs.slice(-60).reverse().map(e => `<tr class="${e[0] === t ? "sel" : ""}" data-act data-t="${e[0]}"><td>${e[0]}</td><td>${e[4].op === "package_shipped" ? "shipped" : "received"}</td><td><span class="link" data-act data-pkg="${esc(e[4].package_id)}">${esc(e[4].package_id)}</span></td><td>${esc(e[4].task_id || "")}</td></tr>`), ["tick", "", "package", "task"]));
  } else if (s.charger != null && R.chargers.some(c => String(c.id) === String(s.charger))){ const ch = R.chargers.find(c => String(c.id) === String(s.charger)), x = resRow("charger", ch.id) || {};
    const sess = R.chargeSess[ch.id] || [], fr = frame(t), now = fr.map((r, i) => r[0] === ch.cell[0] && r[1] === ch.cell[1] && r[6] === R.A.charging ? R.m.robots[i].name + " at " + Math.round(100 * r[2]) + "%" : null).filter(Boolean);
    out.push(selHead(`Charging dock C${ch.id}`, `cell ${ch.key} &middot; ${esc(ch.type)} &middot; ${fmt(ch.power_w)} W`));
    out.push(`<div class="kv"><b>sessions</b><span>${fmt(x.sessions)}${x.open_sessions ? " (" + x.open_sessions + " still open at the end)" : ""}</span><b>robots served</b><span>${fmt(x.robots)}</span><b>occupancy</b><span>${pct(x.occupancy)} (${fmt(x.charging_ticks)} charging ticks)</span><b>energy delivered</b><span>${num(x.energy_wh, 1)} Wh</span><b>mean session</b><span>${num(x.session_ticks_mean, 1)} ticks</span><b>SoC at plug-in &rarr; unplug</b><span>${pct(x.soc_start_mean)} &rarr; ${pct(x.soc_end_mean)}</span><b>stop reasons</b><span>${esc(JSON.stringify(x.stop_reasons || {}))}</span><b>now</b><span>${esc(now.join(", ") || "free")}</span></div>`);
    const B = R.B, use = [], frs = R.run.frames, [cy, cx] = ch.cell;
    for (let k = 0; k < R.N; k += B){ let o = 0, n = 0; for (let j = k; j < Math.min(R.N, k + B); j++, n++) if (frs[j].some(r => r[0] === cy && r[1] === cx && r[6] === R.A.charging)) o++; use.push(o / n); }
    const qser = R.run.frames.map(f => f.filter(r => r[6] === R.A.dock_queue).length);
    charts = [{series:[{name:"dock occupancy (charging)", color:"#3b82f6", data:use, kind:"bar", bucket:B}], ymax:1, yfmt:v => Math.round(v * 100) + "%"},
              {series:[{name:"robots in the dock queue (fleet-wide)", color:"#f59e0b", data:qser, kind:"step", width:1.5}], int:true}];
    out.push("<h4>Sessions (click to jump)</h4>" + (sess.length ? hist(sess.map(z => `<tr class="${z.start <= t && (z.end == null || z.end >= t) ? "sel" : ""}" data-act data-t="${z.start}"><td><span class="link" data-act data-robot="${esc(R.m.robots[z.robot].name)}">${esc(R.m.robots[z.robot].name)}</span></td><td>${z.start}</td><td>${z.end ?? "open"}</td><td>${z.end != null ? z.end - z.start : "&ndash;"}</td><td>${pct(z.soc0)} &rarr; ${z.soc1 == null ? "&ndash;" : pct(z.soc1)}</td><td>${esc(z.reason || "")}</td></tr>`), ["robot", "start", "end", "ticks", "SoC", "reason"]) : '<p class="muted">no charging session at this dock in this run</p>'));
  } else if (s.shelf && R.shelf[s.shelf]){ const v = at(R.shelf[s.shelf], t), pk = Object.entries(R.pkg).filter(([, h]) => { const q = at(h, t); return q && q[1].shelf === s.shelf; }).map(([id]) => id);
    out.push(selHead(`Shelf ${s.shelf}`, v ? "now " + esc(v[1].status) + (v[1].loc ? " at " + esc(v[1].loc) : "") : ""));
    out.push(`<div class="kv"><b>packages on it now</b><span>${pk.map(p => `<span class="link" data-act data-pkg="${p}">${p}</span>`).join(" ") || "none"}</span></div>`);
    out.push("<h4>History (click to jump)</h4>" + hist(R.shelf[s.shelf].map(([tt, x]) => `<tr data-act data-t="${tt}"><td>${tt}</td><td>${esc(x.status)}</td><td>${esc(x.loc || "")}</td><td>${x.cell ? x.cell.join(",") : ""}</td></tr>`), ["tick", "status", "location", "cell"]));
  } else if (s.coal != null){ out.push(selHead(`Simulator task ${esc(s.coal)}`, "not linked to an order task")); }
  else { out.push('<span class="muted">Nothing selected. Click a kanban card, an order, a robot, a package, a decision row, an event, or a station / dock / shelf on the map. Typing an ID in the filters selects it too.</span>'); }
  $("selected").innerHTML = out.filter(x => !x.startsWith("<h4>")).join("");
  $("seldetail").innerHTML = out.filter(x => x.startsWith("<h4>")).join("");
  $("selcharts").style.display = charts ? "" : "none";
  if (charts){ chart("c_sel1", charts[0]); if (charts[1]){ $("c_sel2").style.display = ""; chart("c_sel2", charts[1]); } else { $("c_sel2").style.display = "none"; $("c_sel2_leg").innerHTML = ""; } }
  else { delete CH.c_sel1; delete CH.c_sel2; }
  $("dagbox").style.display = dag ? "" : "none"; if (dag) drawDag(t);
}

function drawDag(t){
  const svg = $("dag"), oid = selOrder();
  if (!oid || !R.orders[oid]){ svg.innerHTML = ""; return; }
  const ids = (R.orders[oid].row.tasks || "").split("|").filter(id => R.tasks[id]), lvl = {};
  const deps = id => ((R.tasks[id]?.row.depends_on) || "").split("|").filter(Boolean);
  const L = id => lvl[id] ?? (lvl[id] = deps(id).length ? 1 + Math.max(...deps(id).map(L)) : 0); ids.forEach(L);
  const byL = {}; ids.forEach(id => (byL[lvl[id]] = byL[lvl[id]] || []).push(id));
  const pos = {}; Object.entries(byL).forEach(([l, arr]) => arr.forEach((id, k) => pos[id] = [4 + l * 215, 8 + k * 34]));
  const h = Math.max(44, 34 * Math.max(1, ...Object.values(byL).map(a => a.length)) + 10); svg.setAttribute("height", h); svg.setAttribute("width", Math.max(...Object.values(pos).map(p => p[0])) + 200);
  let s = '<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6" fill="#6b7280"/></marker></defs>';
  for (const id of ids) for (const d of deps(id)) if (pos[d]) s += `<line x1="${pos[d][0]+188}" y1="${pos[d][1]+12}" x2="${pos[id][0]}" y2="${pos[id][1]+12}" stroke="#6b7280" marker-end="url(#ar)"/>`;
  for (const id of ids){ const st = (taskState(id, t) || [0, "pending"])[1], [x, y] = pos[id];
    s += `<g data-act data-task="${id}" data-order="${oid}"><rect x="${x}" y="${y}" width="188" height="24" rx="5" fill="${SCOL[st]}" opacity="0.9" ${ST.sel.task === id ? 'stroke="#111827" stroke-width="2"' : ""}/><text x="${x+5}" y="${y+16}" fill="#fff" font-size="11">${id} ${esc(R.tasks[id].row.op||"")} &middot; ${st}</text></g>`; }
  svg.innerHTML = s;
}

function summarize(d){ const skip = new Set(["snapshot","locations","shelves","packages","tasks","assigned"]); return Object.entries(d||{}).filter(([k]) => !skip.has(k)).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" "); }
function drawEvents(t){
  const F = filters(), rob = F.robot ? R.m.robots.findIndex(r => r.name === F.robot) : -1, S = ST.sel;
  const so = !F.order && !F.task && S.order ? S.order : null, stc = S.station ? S.station : null, chg = S.charger != null ? String(S.charger) : null;
  const any = F.order || F.pkg || F.task || F.robot || F.coal || F.type || so || stc || chg;
  $("evhint").textContent = so ? "Showing events of the selected order " + so + "." : stc ? "Showing events at the selected station " + stc + "." : chg ? "Showing charging events of dock C" + chg + "." : any ? "Filtered." : "Events within 25 ticks of the current tick (select or filter to see a whole history).";
  const rows = []; for (const e of R.ev){ const [tt, type, robots, task, d] = e;
    if (!any && Math.abs(tt - t) > 25) continue;
    if (F.type && !type.includes(F.type)) continue; if (F.robot && !robots.includes(rob)) continue; if (F.coal && String(task) !== F.coal) continue;
    const ord = F.order || so;
    if (ord && d.order_id !== ord && !(task != null && R.simToTask[task] && R.tasks[R.simToTask[task]].row.order_id === ord)) continue;
    if (F.task && d.task_id !== F.task && R.simToTask[task] !== F.task) continue;
    if (F.pkg && d.package_id !== F.pkg) continue;
    if (stc && !(d.cell && ckey(d.cell) === stc)) continue;
    if (chg && !(String(d.charger) === chg)) continue;
    rows.push(`<tr class="${tt===t?"sel":""}" data-act data-t="${tt}"><td>${tt}</td><td>${esc(type)}${d.op?" &middot; "+esc(d.op):""}</td><td>${esc(robots.map(i => R.m.robots[i].name).join(","))}</td><td>${task ?? ""}</td><td>${esc(summarize(d)).slice(0, 220)}</td></tr>`);
    if (rows.length > 400) break; }
  $("events").innerHTML = `<table><tr><th>tick</th><th>event</th><th>robots</th><th>sim task</th><th>data</th></tr>${rows.join("")}</table>`;
}

function drawMetrics(){
  const m = R.run.metrics; if (!m){ $("metrics").innerHTML = '<p class="muted">no metrics in this trace</p>'; $("diag").innerHTML = ""; return; }
  const kv = o => "<div class='kv'>" + Object.entries(o).filter(([, v]) => typeof v !== "object" || v === null || v.mean !== undefined).map(([k, v]) => `<b>${k}</b><span>${v && v.mean !== undefined ? "mean " + fmt(v.mean) + " &middot; p95 " + fmt(v.p95) : fmt(v)}</span>`).join("") + "</div>";
  $("metrics").innerHTML = "<h2>primary</h2>" + kv(m.primary) + "<h2>order</h2>" + kv(m.order) + "<h2>scheduling</h2>" + kv(m.scheduling) + "<h2>coalition</h2>" + kv(m.coalition) + "<h2>system</h2>" + kv(m.system) +
    "<h2>robot</h2>" + kv(Object.fromEntries(Object.entries(m.robot).filter(([k]) => k !== "per_robot" && k !== "activity_share"))) +
    "<h2>task-time by cause</h2>" + kv(m.bottlenecks.task_share_by_cause);
  const u = m.bottlenecks.unsuccessful_orders;
  $("diag").innerHTML = "<div class='kv'>" + Object.entries(m.bottlenecks.unsuccessful_by_category).map(([k, v]) => `<b>${esc(k)}</b><span>${v}</span>`).join("") + "</div>" +
    `<table><tr><th>order</th><th>state</th><th>age</th><th>diagnosis</th><th>evidence</th></tr>` + u.slice(0, 300).map(x => `<tr data-act data-order="${x.order_id}" data-t="${x.evidence_span ? x.evidence_span[0] : ""}"><td>${x.order_id}</td><td>${x.state}</td><td>${x.age}</td><td>${esc(x.category)}${x.failures.length ? " &middot; " + esc(x.failures[x.failures.length-1].reason) : ""}</td><td>${x.evidence_span ? "ticks " + x.evidence_span[0] + "&ndash;" + x.evidence_span[1] + " " + x.evidence_span[2] : esc(x.reason||"")}</td></tr>`).join("") + "</table>";
}

function redraw(){ const t = tickNow(); $("tlabel").textContent = "tick " + t + " / " + (R.T0 + R.N - 1);
  drawMap(t); drawKanban(t); drawCharts(t); drawOrders(t); drawDecision(t); drawRobots(t); drawPackages(t); drawResources(t); drawGantt(t); drawSelected(t); drawEvents(t); }
function setTick(t){ $("tick").value = Math.max(0, Math.min(R.N - 1, t - R.T0)); redraw(); }
function loadRun(k){ R = prepare(DATA.runs[k]); $("tick").max = R.N - 1; $("tick").value = 0; ST.sel = {}; ST.hover = {}; drawMetrics(); redraw(); }

function csv(rows){ if (!rows || !rows.length) return ""; const keys = [...new Set(rows.flatMap(r => Object.keys(r)))];
  const q = v => { if (v == null) return ""; const s = typeof v === "object" ? JSON.stringify(v) : String(v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  return [keys.join(","), ...rows.map(r => keys.map(k => q(r[k])).join(","))].join("\n"); }
function download(name, text, type){ const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], {type})); a.download = name; a.click(); }
$("export").onchange = () => { const v = $("export").value; $("export").value = ""; if (!v) return; const run = R.run, base = (run.meta.label || "run") + "_" + run.meta.seed + "_";
  if (v === "trace.json") return download(base + v, JSON.stringify(run), "application/json");
  if (v === "metrics.json") return download(base + v, JSON.stringify(run.metrics, null, 1), "application/json");
  if (v === "stations.csv") return download(base + v, csv(run.metrics?.resources?.stations || []), "text/csv");
  if (v === "chargers.csv") return download(base + v, csv(run.metrics?.resources?.chargers || []), "text/csv");
  if (v === "events.csv") return download(base + v, csv(run.events.map(([t, type, robots, task, d]) => ({tick:t, type, robots:robots.join("|"), task, ...Object.fromEntries(Object.entries(d).filter(([k]) => k !== "snapshot"))}))), "text/csv");
  const key = v.replace(".csv", ""); download(base + v, csv(run.tables?.[key] || []), "text/csv"); };

DATA.runs.forEach((r, k) => { const o = document.createElement("option"); o.value = k; o.textContent = (r.meta.label || r.meta.rule || "run") + " · " + r.meta.scenario + " · seed " + r.meta.seed; $("run").appendChild(o); });
$("run").onchange = () => loadRun(Number($("run").value));
$("tick").oninput = redraw;
for (const id of ["f_order","f_pkg","f_task","f_robot","f_coal","f_type"]) $(id).oninput = () => { const F = filters(), s = {};
  if (F.order) s.order = F.order; if (F.task) s.task = F.task; if (F.pkg) s.pkg = F.pkg; if (F.robot) s.robot = F.robot; if (F.coal) s.coal = F.coal; ST.sel = s; redraw(); };
$("clear").onclick = () => { for (const id of ["f_order","f_pkg","f_task","f_robot","f_coal","f_type"]){ $(id).value = ""; $(id).classList.remove("bad"); } select({}); };
$("back").onclick = () => setTick(tickNow() - 1); $("fwd").onclick = () => setTick(tickNow() + 1);
$("play").onclick = () => { playing = !playing; $("play").innerHTML = playing ? "&#10074;&#10074;" : "&#9654;";
  if (playing) timer = setInterval(() => { const t = tickNow() + Number($("speed").value); if (t > R.T0 + R.N - 1){ $("play").click(); return; } setTick(t); }, 120); else clearInterval(timer); };
$("map").addEventListener("pointerdown", ev => { const c = $("map"), r = c.getBoundingClientRect(), cs = c._cs || (c.width / R.m.shape[1]), t = tickNow();
  const x = Math.floor((ev.clientX - r.left) * c.width / r.width / cs), y = Math.floor((ev.clientY - r.top) * c.height / r.height / cs), key = y + "," + x;
  const cand = [];                                          // everything on the cell; clicking again cycles through them
  frame(t).forEach((f, i) => { if (f[0] === y && f[1] === x) cand.push({robot: R.m.robots[i].name}); });
  const ch = R.chargers.find(c2 => c2.key === key); if (ch) cand.push({charger: String(ch.id)});
  if (R.stations.some(s => s.key === key)) cand.push({station: key});
  for (const [s, h] of Object.entries(R.shelf)){ const v = at(h, t); if (v && v[1].status === "stored" && v[1].cell && v[1].cell[0] === y && v[1].cell[1] === x) cand.push({shelf: s}); }
  for (const [s, y2, x2] of (R.run.shelves[t - R.T0] || [])) if (y2 === y && x2 === x) cand.push({shelf: s});
  if (!cand.length) return;
  const same = (a, b) => Object.keys(a).length === Object.keys(b).length && Object.keys(a).every(k => String(a[k]) === String(b[k]));
  const k = cand.findIndex(c2 => same(c2, ST.sel)); select(cand[(k + 1) % cand.length]); });
for (const id of ["c_flow", "c_rate", "c_soc", "c_sel1", "c_sel2"]) hookChart(id);
$("gantt").addEventListener("pointerdown", ev => { const c = $("gantt"), r = c.getBoundingClientRect(), x = (ev.clientX - r.left) * (c.width / r.width);
  if (c._lab && x > c._lab) setTick(Math.round(R.T0 + (x - c._lab) / c._sc)); });
window.onresize = redraw;
const H0 = Object.fromEntries(location.hash.slice(1).split("&").filter(Boolean).map(kv => kv.split("=").map(decodeURIComponent)));
loadRun(Number(H0.run || 0)); $("run").value = String(Number(H0.run || 0));
if (H0.t) setTick(Number(H0.t));
{ const s = {}; for (const k of SELKEYS) if (H0[k] != null) s[k] = H0[k]; if (Object.keys(s).length){ ST.sel = s; redraw(); } }
</script></body></html>
"""


def write_dashboard(traces: Union[Dict, Sequence[Dict]], path: str) -> str:
    runs: List[Dict] = [traces] if isinstance(traces, dict) else list(traces)
    data = json.dumps({"runs": runs}, default=_json_default, separators=(",", ":")).replace("</", "<\\/")
    with open(path, "w") as f:
        f.write(TEMPLATE.replace("__DATA__", data))
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build the offline replay dashboard from trace JSON files.")
    ap.add_argument("traces", nargs="+", help="trace JSON files (TraceRecorder.save / run_episode)")
    ap.add_argument("-o", "--out", default="dashboard.html")
    a = ap.parse_args(argv)
    traces = []
    for p in a.traces:
        with open(p) as f:
            traces.append(json.load(f))
    print(write_dashboard(traces, a.out))


if __name__ == "__main__":
    main()
