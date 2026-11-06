"""Offline, self-contained HTML replay dashboard (no server, no external libraries).

    python -m battery_tarware.orders.dashboard trace_fifo.json trace_plb.json -o dashboard.html

Open it in a browser; ``dashboard.html#run=1&t=300&order=O00012`` opens run 1 at tick 300 with an
order selected.

or ``write_dashboard([trace, ...], "dashboard.html")`` with traces from ``run_episode``. One file
holds the recorded runs; everything shown is reconstructed from each run's single trace (event log
and per-tick frames), so the dashboard, CSV/JSON exports and metrics agree.

Panels, synchronised on the selected tick: warehouse map (zones, shelves, robots with battery,
carried shelves, selected crew), order-task **kanban** (one column per lifecycle state), order queue,
dependency graph of the selected order, the scheduler decision of the tick (candidates, scores,
feasible coalitions, outcomes), robot and package tables, battery and backlog charts, a per-task
cause timeline (bottleneck attribution), metrics and unsuccessful-order diagnosis (each with an
evidence span that jumps to its tick), and the event timeline. Filters: run (policy), order,
package, task, robot, coalition (simulator task) and event type. Exports: CSV ledgers, events CSV,
trace JSON, metrics JSON.
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
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d2330;--muted:#6b7280;--line:#e3e6eb;--acc:#2563eb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.35 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);padding:8px 14px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header h1{font-size:15px;margin:0 8px 0 0}header select,header input,header button{font:inherit;padding:3px 6px;border:1px solid var(--line);border-radius:6px;background:#fff}
header button{cursor:pointer}#tick{width:min(520px,60vw)}.filters input{width:84px}
main{display:grid;grid-template-columns:minmax(320px,1.05fr) minmax(320px,1fr);gap:10px;padding:10px}
@media(max-width:900px){main{grid-template-columns:1fr}}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 10px;min-width:0}
section h2{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:2px 0 6px}
.wide{grid-column:1/-1}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:2px 5px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:600}.scroll{max-height:260px;overflow:auto}tr.sel td{background:#e8f0ff}tr.click{cursor:pointer}tr.click:hover td{background:#f2f5fa}
#kanban{display:grid;grid-template-columns:repeat(10,minmax(86px,1fr));gap:6px;overflow-x:auto}
.col{background:#f2f4f7;border-radius:8px;padding:4px;min-height:80px}.col h3{font-size:11px;margin:2px 2px 4px;color:var(--muted);display:flex;justify-content:space-between}
.card{background:#fff;border:1px solid var(--line);border-left:4px solid #999;border-radius:6px;padding:3px 5px;margin-bottom:4px;font-size:11px;cursor:pointer}
.card.sel{outline:2px solid var(--acc)}.card small{color:var(--muted);display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pill{display:inline-block;padding:0 6px;border-radius:9px;font-size:11px;background:#eef}.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px}
.kv b{font-weight:600}canvas{max-width:100%;display:block}.muted{color:var(--muted)}.bar{height:8px;background:#e5e7eb;border-radius:4px;overflow:hidden;width:60px;display:inline-block;vertical-align:middle}
.bar i{display:block;height:100%}.legend span{display:inline-block;margin-right:8px;font-size:11px}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:3px;vertical-align:-1px}
</style></head><body>
<header>
 <h1>Battery-TA-RWARE replay</h1>
 <label>run <select id="run"></select></label>
 <button id="play">&#9654;</button><button id="back">&#8722;1</button><button id="fwd">+1</button>
 <input id="tick" type="range" min="0" max="0" value="0"><span id="tlabel" class="pill"></span>
 <label>speed <select id="speed"><option>1</option><option selected>4</option><option>16</option><option>64</option></select></label>
 <span class="filters">order <input id="f_order" placeholder="O00012"> package <input id="f_pkg" placeholder="P00003">
 task <input id="f_task" placeholder="T00007"> robot <input id="f_robot" placeholder="agv_2"> coalition <input id="f_coal" placeholder="sim task id">
 event <input id="f_type" placeholder="type"></span>
 <span>export <select id="export"><option value="">choose&hellip;</option><option>orders.csv</option><option>tasks.csv</option><option>packages.csv</option>
 <option>shelves.csv</option><option>robots.csv</option><option>trips.csv</option><option>events.csv</option><option>metrics.json</option><option>trace.json</option></select></span>
</header>
<main>
 <section><h2>Warehouse</h2><canvas id="map"></canvas><div id="maplegend" class="legend"></div></section>
 <section><h2>Selected</h2><div id="selected" class="muted">Click a card, robot, order, event or package.</div>
   <h2 style="margin-top:8px">Dependency graph (selected order)</h2><svg id="dag" width="100%" height="120"></svg></section>
 <section class="wide"><h2>Order tasks &mdash; kanban at this tick</h2><div id="kanban"></div></section>
 <section><h2>Scheduler decision at this tick</h2><div id="decision" class="scroll"></div></section>
 <section><h2>Order queue (open)</h2><div id="orders" class="scroll"></div></section>
 <section><h2>Robots</h2><div id="robots" class="scroll"></div></section>
 <section><h2>Packages</h2><div id="packages" class="scroll"></div></section>
 <section><h2>Battery (state of charge)</h2><canvas id="soc" height="150"></canvas></section>
 <section><h2>Backlog and completions</h2><canvas id="backlog" height="150"></canvas></section>
 <section class="wide"><h2>Cause timeline (bottleneck attribution per task)</h2><div id="causelegend" class="legend"></div><canvas id="gantt" height="160"></canvas></section>
 <section><h2>Metrics</h2><div id="metrics" class="scroll" style="max-height:340px"></div></section>
 <section><h2>Unsuccessful orders &mdash; diagnosis</h2><div id="diag" class="scroll" style="max-height:340px"></div></section>
 <section class="wide"><h2>Event timeline</h2><div id="events" class="scroll" style="max-height:300px"></div></section>
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
const ZCOL = {aisle:"#ffffff",storage:"#f3ead8",goal:"#d1fae5",dock:"#dbeafe"};
let R, ST = {sel:{}}, playing = false, timer = null;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = v => v == null ? "&ndash;" : (typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : esc(typeof v === "object" ? JSON.stringify(v) : v));
function at(hist, t){ let v = null; for (const h of hist){ if (h[0] <= t) v = h; else break; } return v; }

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
  for (const e of ev){
    const [t, type, robots, task, d] = e;
    if (type === "order_task"){ (P.tasks[d.task_id] = P.tasks[d.task_id] || {row:{task_id:d.task_id, order_id:d.order_id, op:d.op}, hist:[]}).hist.push([t, d.to, d.reason, d.attempt]); if (task != null) P.simToTask[task] = d.task_id; }
    else if (type === "schedule_decision") P.decisions[t] = d;
    else if (type === "inventory"){
      const op = d.op;
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
  P.taskIds = Object.keys(P.tasks).sort();
  return P;
}

function tickNow(){ return R.T0 + Number($("tick").value); }
function frame(t){ return R.run.frames[Math.max(0, Math.min(R.N-1, t - R.T0))]; }
function taskState(id, t){ const h = at(R.tasks[id].hist, t); return h ? h : null; }
function filters(){ return {order:$("f_order").value.trim(), pkg:$("f_pkg").value.trim(), task:$("f_task").value.trim(), robot:$("f_robot").value.trim(), coal:$("f_coal").value.trim(), type:$("f_type").value.trim()}; }
function taskMatches(id, F){ const r = R.tasks[id].row;
  if (F.order && r.order_id !== F.order) return false; if (F.task && id !== F.task) return false;
  if (F.pkg && r.package_id !== F.pkg) return false; if (F.coal && !(r.sim_tasks||"").split("|").includes(F.coal)) return false; return true; }

function drawMap(t){
  const c = $("map"), m = R.m, [H, W] = m.shape, cs = Math.max(8, Math.floor(Math.min(560, c.parentElement.clientWidth - 20) / W));
  c.width = W*cs; c.height = H*cs; const g = c.getContext("2d");
  g.fillStyle = "#374151"; g.fillRect(0,0,c.width,c.height);
  for (const [z, cells] of Object.entries(m.zones)) for (const [y,x] of cells){ g.fillStyle = ZCOL[z] || "#e5e7eb"; g.fillRect(x*cs, y*cs, cs-1, cs-1); }
  for (const ch of m.chargers){ g.fillStyle = "#93c5fd"; g.fillRect(ch.cell[1]*cs+2, ch.cell[0]*cs+2, cs-5, cs-5); }
  for (const [s, h] of Object.entries(R.shelf)){ const v = at(h, t); if (v && v[1].status === "stored" && v[1].cell){ const [y,x] = v[1].cell;
      const full = v[1] && Object.values(R.pkg).some(ph => { const q = at(ph, t); return q && q[1].shelf === s && q[1].status === "stored"; });
      g.fillStyle = full ? "#a16207" : "#d6b98c"; g.fillRect(x*cs+2, y*cs+2, cs-5, cs-5); } }
  const fr = frame(t), selCrew = new Set(); const selT = ST.sel.task && R.tasks[ST.sel.task];
  const selSim = selT ? Number(((selT.row.sim_tasks||"").split("|").pop())) : (ST.sel.coal != null ? Number(ST.sel.coal) : null);
  fr.forEach((r, i) => { if (selSim != null && r[4] === selSim) selCrew.add(i); });
  for (const [s, y, x, st] of (R.run.shelves[t - R.T0] || [])){ g.fillStyle = st === "dropped" ? "#f97316" : "#a16207"; g.fillRect(x*cs+1, y*cs+1, cs-3, cs-3); }
  fr.forEach((r, i) => { const [y, x, soc, status, task] = r, rb = R.m.robots[i];
    g.beginPath(); g.arc(x*cs+cs/2, y*cs+cs/2, cs*0.34, 0, 7); g.fillStyle = status >= 3 ? "#111827" : TYPECOL[R.types.indexOf(rb.type) % 8]; g.fill();
    if (selCrew.has(i) || ST.sel.robot === rb.name){ g.lineWidth = 3; g.strokeStyle = "#facc15"; g.stroke(); }
    g.fillStyle = soc < 0.3 ? "#dc2626" : "#16a34a"; g.fillRect(x*cs+2, y*cs+cs-4, Math.max(1,(cs-5)*soc), 3); });
  $("maplegend").innerHTML = R.types.map((ty, k) => `<span><i style="background:${TYPECOL[k%8]}"></i>${esc(ty)}</span>`).join("") +
    '<span><i style="background:#a16207"></i>shelf with package</span><span><i style="background:#d6b98c"></i>empty shelf</span><span><i style="background:#93c5fd"></i>dock</span><span><i style="background:#d1fae5"></i>station</span><span><i style="background:#111827"></i>broken/stranded</span>';
}

function drawKanban(t){
  const F = filters(), cols = {}; STATES.forEach(s => cols[s] = []);
  for (const id of R.taskIds){ if (!taskMatches(id, F)) continue; const h = taskState(id, t); if (!h) continue; cols[h[1]].push([id, h]); }
  cols.completed = cols.completed.sort((a,b) => b[1][0]-a[1][0]).slice(0, 12); cols.cancelled = cols.cancelled.slice(-12);
  $("kanban").innerHTML = STATES.map(s => `<div class="col"><h3><span>${s}</span><span>${cols[s].length}</span></h3>` + cols[s].slice(0, 40).map(([id, h]) => {
      const r = R.tasks[id].row; return `<div class="card${ST.sel.task===id?" sel":""}" style="border-left-color:${SCOL[s]}" data-task="${id}"><b>${id}</b> ${esc(r.op||"")}<small>${esc(r.order_id)} ${esc(r.package_id||r.shelf_id||"")}${h[3]>1?" &middot; try "+h[3]:""}</small>${h[2]?`<small title="${esc(h[2])}">${esc(h[2])}</small>`:""}</div>`; }).join("") + "</div>").join("");
  for (const el of document.querySelectorAll(".card")) el.onclick = () => select({task: el.dataset.task});
}

function drawOrders(t){
  const F = filters(), rows = [];
  for (const [id, o] of Object.entries(R.orders)){ if (o.arrival > t || (o.closed != null && o.closed <= t)) continue; if (F.order && id !== F.order) continue;
    const tids = (o.row.tasks||"").split("|").filter(Boolean), done = tids.filter(x => (taskState(x, t)||[])[1] === "completed").length;
    rows.push(`<tr class="click${ST.sel.order===id?" sel":""}" data-order="${id}"><td>${id}</td><td>${esc(o.row.kind)}</td><td>${t-o.arrival}</td><td>${fmt(o.row.priority)}</td><td>${o.row.due==null?"&ndash;":o.row.due-t}</td><td>${done}/${tids.length}</td></tr>`); }
  $("orders").innerHTML = `<table><tr><th>order</th><th>kind</th><th>age</th><th>prio</th><th>due in</th><th>tasks done</th></tr>${rows.join("")}</table>` + (rows.length ? "" : '<p class="muted">no open orders</p>');
  for (const el of $("orders").querySelectorAll("tr[data-order]")) el.onclick = () => select({order: el.dataset.order});
}

function drawDecision(t){
  const d = R.decisions[t]; if (!d){ $("decision").innerHTML = '<p class="muted">no decision record at this tick (no eligible task, or a coordinator without a decision log)</p>'; return; }
  const rows = d.tasks.map(r => `<tr class="click" data-coal="${r[0]}"><td>${r[0]}</td><td>${esc(r[1]||"")}</td><td>${fmt(r[2])}</td><td>${r[3]}</td><td>${fmt(r[4])}</td><td>${esc(r[5])}</td><td>${esc((d.assigned[String(r[0])]||[]).map(i => R.m.robots[i].name).join(", "))}</td></tr>`);
  $("decision").innerHTML = `<div class="kv"><b>rule</b><span>${esc(d.rule)}</span><b>free robots</b><span>${d.free}</span><b>candidate pairs</b><span>${d.pairs}</span></div>
   <table><tr><th>sim task</th><th>task</th><th>score (sorts first)</th><th>coalitions</th><th>best est.</th><th>outcome</th><th>crew</th></tr>${rows.join("")}</table>`;
  for (const el of $("decision").querySelectorAll("tr[data-coal]")) el.onclick = () => { const id = R.simToTask[el.dataset.coal]; select(id ? {task:id} : {coal:el.dataset.coal}); };
}

function drawRobots(t){
  const fr = frame(t), F = filters(), m = R.m;
  const rows = fr.map((r, i) => { const rb = m.robots[i]; if (F.robot && rb.name !== F.robot) return "";
    const task = r[4] >= 0 ? (R.simToTask[r[4]] || "sim "+r[4]) : "";
    return `<tr class="click${ST.sel.robot===rb.name?" sel":""}" data-robot="${rb.name}"><td>${esc(rb.name)}</td><td>${esc(m.activities[r[6]])}</td><td><span class="bar"><i style="width:${Math.round(100*r[2])}%;background:${r[2]<0.3?"#dc2626":"#16a34a"}"></i></span> ${Math.round(100*r[2])}%</td><td>${esc(task)}</td><td>${r[5]>=0?esc(m.roles[r[5]]):""}</td><td>${r[0]},${r[1]}</td></tr>`; });
  $("robots").innerHTML = `<table><tr><th>robot</th><th>activity</th><th>battery</th><th>task</th><th>role</th><th>cell</th></tr>${rows.join("")}</table>`;
  for (const el of $("robots").querySelectorAll("tr[data-robot]")) el.onclick = () => select({robot: el.dataset.robot});
}

function drawPackages(t){
  const F = filters(), rows = [];
  for (const id of Object.keys(R.pkg).sort()){ if (F.pkg && id !== F.pkg) continue; const v = at(R.pkg[id], t); if (!v) continue; const p = v[1];
    if (!F.pkg && (p.status === "shipped" || p.status === "cancelled") && v[0] < t - 50) continue;
    rows.push(`<tr class="click${ST.sel.pkg===id?" sel":""}" data-pkg="${id}"><td>${id}</td><td>${esc(p.status)}</td><td>${esc(p.shelf||"")}</td><td>${esc(p.loc||"")}</td><td>${esc(p.reserved||"")}</td><td>${fmt(p.weight)}</td></tr>`); }
  $("packages").innerHTML = `<table><tr><th>package</th><th>status</th><th>shelf</th><th>location</th><th>reserved by</th><th>kg</th></tr>${rows.join("")}</table>`;
  for (const el of $("packages").querySelectorAll("tr[data-pkg]")) el.onclick = () => select({pkg: el.dataset.pkg});
}

function lineChart(id, series, colors, t, ymax){
  const c = $(id), w = c.parentElement.clientWidth - 20; c.width = w; const h = c.height, g = c.getContext("2d"); g.clearRect(0,0,w,h);
  const n = R.N, step = Math.max(1, Math.floor(n / w)); let mx = ymax || 1; if (!ymax) for (const s of series) for (const v of s) if (v > mx) mx = v;
  g.strokeStyle = "#e5e7eb"; g.beginPath(); g.moveTo(0,h-1); g.lineTo(w,h-1); g.stroke();
  series.forEach((s, k) => { g.strokeStyle = colors[k % colors.length]; g.lineWidth = 1; g.beginPath(); let first = true;
    for (let i = 0; i < n; i += step){ const v = s[i]; if (v == null) continue; const x = i / n * w, y = h - 4 - v / mx * (h - 10); if (first){ g.moveTo(x,y); first = false; } else g.lineTo(x,y); } g.stroke(); });
  const cx = (t - R.T0) / n * w; g.strokeStyle = "#111"; g.beginPath(); g.moveTo(cx, 0); g.lineTo(cx, h); g.stroke();
  g.fillStyle = "#6b7280"; g.fillText(String(mx === 1 && ymax ? "100%" : Math.round(mx)), 2, 10);
}
function drawCharts(t){
  const fr = R.run.frames, socs = R.m.robots.map((_, i) => fr.map(f => f[i][2]));
  lineChart("soc", socs, R.m.robots.map(r => TYPECOL[R.types.indexOf(r.type) % 8]), t, 1);
  const s = R.run.series; lineChart("backlog", [s.open_orders || [], s.completed_orders || []], ["#f59e0b", "#059669"], t);
}

function drawGantt(t){
  const c = $("gantt"), w = c.parentElement.clientWidth - 20, F = filters(); c.width = w;
  let ids = R.taskIds.filter(id => taskMatches(id, F) && R.run.spans[id]);
  if (ST.sel.order) ids = ids.filter(id => R.tasks[id].row.order_id === ST.sel.order);
  else if (!F.order && !F.task && !F.pkg && !F.coal) ids = ids.filter(id => R.run.spans[id].some(s => s[0] <= t && s[1] >= t - 30)).slice(0, 40);
  const rh = 12, h = Math.max(40, ids.length * rh + 14); c.height = h; const g = c.getContext("2d"); g.clearRect(0,0,w,h);
  const lab = 64, sc = (w - lab) / R.N;
  ids.forEach((id, k) => { g.fillStyle = "#374151"; g.font = "10px sans-serif"; g.fillText(id, 2, k*rh + 10);
    for (const [a, b, cause] of R.run.spans[id]){ g.fillStyle = CCOL[cause] || "#999"; g.fillRect(lab + (a - R.T0) * sc, k*rh + 2, Math.max(1, (b - a + 1) * sc), rh - 3); } });
  const cx = lab + (t - R.T0) * sc; g.strokeStyle = "#111"; g.beginPath(); g.moveTo(cx, 0); g.lineTo(cx, h); g.stroke();
  c.onclick = ev => { const r = c.getBoundingClientRect(), x = (ev.clientX - r.left) * (c.width / r.width); if (x > lab) setTick(Math.round(R.T0 + (x - lab) / sc)); };
  $("causelegend").innerHTML = Object.entries(CCOL).map(([k, v]) => `<span><i style="background:${v}"></i>${k}</span>`).join("");
}

function drawSelected(t){
  const s = ST.sel, out = [];
  if (s.task && R.tasks[s.task]){ const T = R.tasks[s.task], r = T.row;
    out.push(`<div class="kv"><b>task</b><span>${s.task} (${esc(r.op)}) of ${esc(r.order_id)}</span><b>package / shelf</b><span>${esc(r.package_id||"")} / ${esc(r.shelf_id||"")}</span><b>depends on</b><span>${esc(r.depends_on||"")}</span><b>attempts</b><span>${fmt(r.attempt)} (sim ${esc(r.sim_tasks||"")})</span><b>last failure</b><span>${esc(r.failure_reason||"")} ${r.blocking_resource?"&middot; "+esc(r.blocking_resource):""}</span></div>`);
    out.push("<table><tr><th>tick</th><th>state</th><th>reason</th></tr>" + T.hist.map(h => `<tr class="click" data-t="${h[0]}"><td>${h[0]}</td><td>${h[1]}</td><td>${esc(h[2]||"")}</td></tr>`).join("") + "</table>");
  } else if (s.pkg && R.pkg[s.pkg]){
    out.push(`<b>${s.pkg}</b><table><tr><th>tick</th><th>status</th><th>shelf</th><th>location</th><th>reserved</th></tr>` + R.pkg[s.pkg].map(([tt, p]) => `<tr class="click" data-t="${tt}"><td>${tt}</td><td>${esc(p.status)}</td><td>${esc(p.shelf||"")}</td><td>${esc(p.loc||"")}</td><td>${esc(p.reserved||"")}</td></tr>`).join("") + "</table>");
  } else if (s.robot){ const i = R.m.robots.findIndex(r => r.name === s.robot), rb = R.m.robots[i];
    out.push(`<div class="kv"><b>robot</b><span>${esc(rb.name)} (${esc(rb.type)})</span><b>capabilities</b><span>${esc(rb.capabilities.join(", "))}</span><b>payload</b><span>${rb.payload_kg} kg</span><b>battery</b><span>${rb.capacity_wh} Wh</span></div>`);
    const pr = (R.run.metrics?.robot?.per_robot || [])[i]; if (pr) out.push("<div class='kv'>" + Object.entries(pr).map(([k, v]) => `<b>${k}</b><span>${fmt(v)}</span>`).join("") + "</div>");
  } else if (s.order && R.orders[s.order]){ const o = R.orders[s.order].row;
    out.push("<div class='kv'>" + Object.entries(o).map(([k, v]) => `<b>${k}</b><span>${fmt(v)}</span>`).join("") + "</div>");
  } else { out.push('<span class="muted">Click a card, robot, order, event or package.</span>'); }
  $("selected").innerHTML = out.join("");
  for (const el of $("selected").querySelectorAll("tr[data-t]")) el.onclick = () => setTick(Number(el.dataset.t));
  drawDag(t);
}

function drawDag(t){
  const svg = $("dag"), oid = ST.sel.order || (ST.sel.task && R.tasks[ST.sel.task] ? R.tasks[ST.sel.task].row.order_id : null);
  if (!oid || !R.orders[oid]){ svg.innerHTML = '<text x="4" y="16" fill="#6b7280" font-size="12">select an order or task</text>'; return; }
  const ids = (R.orders[oid].row.tasks || "").split("|").filter(Boolean), lvl = {};
  const deps = id => ((R.tasks[id]?.row.depends_on) || "").split("|").filter(Boolean);
  const L = id => lvl[id] ?? (lvl[id] = deps(id).length ? 1 + Math.max(...deps(id).map(L)) : 0); ids.forEach(L);
  const byL = {}; ids.forEach(id => (byL[lvl[id]] = byL[lvl[id]] || []).push(id));
  const pos = {}; Object.entries(byL).forEach(([l, arr]) => arr.forEach((id, k) => pos[id] = [10 + l * 120, 12 + k * 36]));
  const h = Math.max(60, 36 * Math.max(...Object.values(byL).map(a => a.length)) + 10); svg.setAttribute("height", h);
  let s = '<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6" fill="#6b7280"/></marker></defs>';
  for (const id of ids) for (const d of deps(id)) if (pos[d]) s += `<line x1="${pos[d][0]+96}" y1="${pos[d][1]+12}" x2="${pos[id][0]}" y2="${pos[id][1]+12}" stroke="#6b7280" marker-end="url(#ar)"/>`;
  for (const id of ids){ const st = (taskState(id, t) || [0, "pending"])[1], [x, y] = pos[id];
    s += `<g style="cursor:pointer" data-task="${id}"><rect x="${x}" y="${y}" width="96" height="24" rx="5" fill="${SCOL[st]}" opacity="0.9"/><text x="${x+5}" y="${y+16}" fill="#fff" font-size="11">${id} ${esc(R.tasks[id].row.op||"")}</text></g>`; }
  svg.innerHTML = s; for (const el of svg.querySelectorAll("g[data-task]")) el.onclick = () => select({task: el.dataset.task, order: oid});
}

function summarize(d){ const skip = new Set(["snapshot","locations","shelves","packages","tasks","assigned"]); return Object.entries(d||{}).filter(([k]) => !skip.has(k)).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join(" "); }
function drawEvents(t){
  const F = filters(), rob = F.robot ? R.m.robots.findIndex(r => r.name === F.robot) : -1, any = F.order || F.pkg || F.task || F.robot || F.coal || F.type;
  const rows = []; for (const e of R.ev){ const [tt, type, robots, task, d] = e;
    if (!any && Math.abs(tt - t) > 25) continue;
    if (F.type && !type.includes(F.type)) continue; if (F.robot && !robots.includes(rob)) continue; if (F.coal && String(task) !== F.coal) continue;
    if (F.order && d.order_id !== F.order && !(task != null && R.simToTask[task] && R.tasks[R.simToTask[task]].row.order_id === F.order)) continue;
    if (F.task && d.task_id !== F.task && R.simToTask[task] !== F.task) continue;
    if (F.pkg && d.package_id !== F.pkg) continue;
    rows.push(`<tr class="click${tt===t?" sel":""}" data-t="${tt}"><td>${tt}</td><td>${esc(type)}${d.op?" &middot; "+esc(d.op):""}</td><td>${esc(robots.map(i => R.m.robots[i].name).join(","))}</td><td>${task ?? ""}</td><td>${esc(summarize(d)).slice(0, 220)}</td></tr>`);
    if (rows.length > 400) break; }
  $("events").innerHTML = `<table><tr><th>tick</th><th>event</th><th>robots</th><th>sim task</th><th>data</th></tr>${rows.join("")}</table>`;
  for (const el of $("events").querySelectorAll("tr[data-t]")) el.onclick = () => setTick(Number(el.dataset.t));
}

function drawMetrics(){
  const m = R.run.metrics; if (!m){ $("metrics").innerHTML = '<p class="muted">no metrics in this trace</p>'; $("diag").innerHTML = ""; return; }
  const kv = o => "<div class='kv'>" + Object.entries(o).filter(([, v]) => typeof v !== "object" || v === null || v.mean !== undefined).map(([k, v]) => `<b>${k}</b><span>${v && v.mean !== undefined ? "mean " + fmt(v.mean) + " &middot; p95 " + fmt(v.p95) : fmt(v)}</span>`).join("") + "</div>";
  $("metrics").innerHTML = "<h2>primary</h2>" + kv(m.primary) + "<h2>order</h2>" + kv(m.order) + "<h2>scheduling</h2>" + kv(m.scheduling) + "<h2>coalition</h2>" + kv(m.coalition) + "<h2>system</h2>" + kv(m.system) +
    "<h2>robot</h2>" + kv(Object.fromEntries(Object.entries(m.robot).filter(([k]) => k !== "per_robot" && k !== "activity_share"))) +
    "<h2>task-time by cause</h2>" + kv(m.bottlenecks.task_share_by_cause);
  const u = m.bottlenecks.unsuccessful_orders;
  $("diag").innerHTML = "<div class='kv'>" + Object.entries(m.bottlenecks.unsuccessful_by_category).map(([k, v]) => `<b>${esc(k)}</b><span>${v}</span>`).join("") + "</div>" +
    `<table><tr><th>order</th><th>state</th><th>age</th><th>diagnosis</th><th>evidence</th></tr>` + u.slice(0, 300).map(x => `<tr class="click" data-order="${x.order_id}" data-t="${x.evidence_span ? x.evidence_span[0] : ""}"><td>${x.order_id}</td><td>${x.state}</td><td>${x.age}</td><td>${esc(x.category)}${x.failures.length ? " &middot; " + esc(x.failures[x.failures.length-1].reason) : ""}</td><td>${x.evidence_span ? "ticks " + x.evidence_span[0] + "&ndash;" + x.evidence_span[1] + " " + x.evidence_span[2] : esc(x.reason||"")}</td></tr>`).join("") + "</table>";
  for (const el of $("diag").querySelectorAll("tr[data-order]")) el.onclick = () => { select({order: el.dataset.order}); if (el.dataset.t) setTick(Number(el.dataset.t)); };
}

function redraw(){ const t = tickNow(); $("tlabel").textContent = "tick " + t + " / " + (R.T0 + R.N - 1);
  drawMap(t); drawKanban(t); drawOrders(t); drawDecision(t); drawRobots(t); drawPackages(t); drawCharts(t); drawGantt(t); drawSelected(t); drawEvents(t); }
function setTick(t){ $("tick").value = Math.max(0, Math.min(R.N - 1, t - R.T0)); redraw(); }
function select(s){ ST.sel = s; redraw(); }
function loadRun(k){ R = prepare(DATA.runs[k]); $("tick").max = R.N - 1; $("tick").value = 0; ST.sel = {}; drawMetrics(); redraw(); }

function csv(rows){ if (!rows || !rows.length) return ""; const keys = [...new Set(rows.flatMap(r => Object.keys(r)))];
  const q = v => { if (v == null) return ""; const s = typeof v === "object" ? JSON.stringify(v) : String(v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  return [keys.join(","), ...rows.map(r => keys.map(k => q(r[k])).join(","))].join("\n"); }
function download(name, text, type){ const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], {type})); a.download = name; a.click(); }
$("export").onchange = () => { const v = $("export").value; $("export").value = ""; if (!v) return; const run = R.run, base = (run.meta.label || "run") + "_" + run.meta.seed + "_";
  if (v === "trace.json") return download(base + v, JSON.stringify(run), "application/json");
  if (v === "metrics.json") return download(base + v, JSON.stringify(run.metrics, null, 1), "application/json");
  if (v === "events.csv") return download(base + v, csv(run.events.map(([t, type, robots, task, d]) => ({tick:t, type, robots:robots.join("|"), task, ...Object.fromEntries(Object.entries(d).filter(([k]) => k !== "snapshot"))}))), "text/csv");
  const key = v.replace(".csv", ""); download(base + v, csv(run.tables?.[key] || []), "text/csv"); };

DATA.runs.forEach((r, k) => { const o = document.createElement("option"); o.value = k; o.textContent = (r.meta.label || r.meta.rule || "run") + " · " + r.meta.scenario + " · seed " + r.meta.seed; $("run").appendChild(o); });
$("run").onchange = () => loadRun(Number($("run").value));
$("tick").oninput = redraw;
for (const id of ["f_order","f_pkg","f_task","f_robot","f_coal","f_type"]) $(id).oninput = () => { ST.sel = {}; const F = filters();
  if (F.order) ST.sel.order = F.order; if (F.task) ST.sel.task = F.task; if (F.pkg) ST.sel.pkg = F.pkg; if (F.robot) ST.sel.robot = F.robot; if (F.coal) ST.sel.coal = F.coal; redraw(); };
$("back").onclick = () => setTick(tickNow() - 1); $("fwd").onclick = () => setTick(tickNow() + 1);
$("play").onclick = () => { playing = !playing; $("play").innerHTML = playing ? "&#10074;&#10074;" : "&#9654;";
  if (playing) timer = setInterval(() => { const t = tickNow() + Number($("speed").value); if (t > R.T0 + R.N - 1){ $("play").click(); return; } setTick(t); }, 120); else clearInterval(timer); };
$("map").onclick = ev => { const c = $("map"), r = c.getBoundingClientRect(), cs = c.width / R.m.shape[1];
  const x = Math.floor((ev.clientX - r.left) * c.width / r.width / cs), y = Math.floor((ev.clientY - r.top) * c.height / r.height / cs);
  const i = frame(tickNow()).findIndex(f => f[0] === y && f[1] === x); if (i >= 0) select({robot: R.m.robots[i].name}); };
window.onresize = redraw;
const H0 = Object.fromEntries(location.hash.slice(1).split("&").filter(Boolean).map(kv => kv.split("=")));
loadRun(Number(H0.run || 0)); $("run").value = String(Number(H0.run || 0));
if (H0.t) setTick(Number(H0.t)); if (H0.task) select({task: H0.task}); if (H0.order) select({order: H0.order});
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
