// Browser-side test plan for the replay dashboard (injected by experiments/checks/dashboard_browser.py).
// Reads window.__BT_PLAN, drives the page through the same handlers a user triggers, and writes the
// observations as JSON into <pre id="BT_RESULT">. No assertions here: Python compares the
// observations with the trace (event log and frames).
(function () {
  const OUT = {errors: [], kanban: {}, orders_open: {}, robots: {}, decisions: {}, downloads: [], steps: []};
  window.addEventListener("error", e => OUT.errors.push(String(e.message) + " @" + e.lineno));
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const press = (el, x, y) => el.dispatchEvent(new PointerEvent("pointerdown", {bubbles: true, button: 0, clientX: x || 0, clientY: y || 0}));
  const mapPress = cell => { const c = $("map"), r = c.getBoundingClientRect(), cs = c._cs * r.width / c.width;
    press(c, r.left + (cell[1] + .5) * cs, r.top + (cell[0] + .5) * cs); };
  const kanbanNow = () => { const o = {counts: {}, cards: {}};
    for (const col of document.querySelectorAll("#kanban .col")) { const [name, n] = [...col.querySelectorAll("h3 span")].map(s => s.textContent);
      o.counts[name] = Number(n); for (const c of col.querySelectorAll(".card")) o.cards[c.dataset.task] = name; }
    return o; };
  async function run() {
    const P = window.__BT_PLAN || {};
    try {
      download = (name, text) => OUT.downloads.push({name, lines: text ? text.split("\n").length : 0, head: (text || "").slice(0, 200)});
      OUT.meta = {T0: R.T0, N: R.N, robots: R.m.robots.length, hash: location.hash};
      OUT.initial_sel = JSON.parse(JSON.stringify(ST.sel)); OUT.initial_tick = tickNow();
      if (P.deep_link_only) return;
      for (const t of P.ticks || []) {
        setTick(t);
        OUT.kanban[t] = kanbanNow();
        OUT.orders_open[t] = document.querySelectorAll("#orders tr[data-order]").length;
        OUT.robots[t] = [...document.querySelectorAll("#robots tr[data-robot]")].map(tr => [tr.dataset.robot, tr.children[1].textContent, tr.children[2].textContent.trim()]);
        OUT.decisions[t] = [...document.querySelectorAll("#decision tr[data-coal]")].map(tr => [Number(tr.dataset.coal), tr.children[5].textContent]);
      }
      // playback
      setTick(P.play_from); $("speed").value = "4"; $("play").click(); await sleep(1500); $("play").click();
      OUT.playback = {from: P.play_from, to: tickNow()};
      // selection during playback: press a card while playing
      $("play").click(); await sleep(300);
      const card = document.querySelector("#kanban .card");
      if (card) { press(card); OUT.play_select = {task: card.dataset.task, sel: ST.sel.task, head: ($("selected").querySelector(".selhead") || {}).textContent}; }
      $("play").click();
      // order selection, dependency graph
      if (P.dag_order) {
        setTick(P.dag_tick); select({order: P.dag_order});
        OUT.dag = {nodes: [...$("dag").querySelectorAll("g[data-task]")].map(g => g.dataset.task), edges: $("dag").querySelectorAll("line").length,
                   rows: [...$("seldetail").querySelectorAll("tr[data-task]")].map(tr => [tr.dataset.task, tr.children[2].textContent]),
                   highlighted: [...document.querySelectorAll("#kanban .card.of, #kanban .card.sel")].map(c => c.dataset.task)};
        const node = $("dag").querySelector("g[data-task]"); press(node); OUT.dag.node_select = ST.sel;
      }
      // filter by short id
      if (P.filter_order) { $("f_order").value = P.filter_order.short; $("f_order").oninput(); setTick(P.filter_order.tick);
        OUT.filter = {sel: ST.sel, bad: $("f_order").classList.contains("bad"), orders_rows: [...document.querySelectorAll("#orders tr[data-order]")].map(tr => tr.dataset.order),
                      cards: [...document.querySelectorAll("#kanban .card")].map(c => R.tasks[c.dataset.task].row.order_id),
                      events: document.querySelectorAll("#events tr[data-t]").length, hint: $("evhint").textContent};
        $("f_order").value = "nonexistent"; $("f_order").oninput(); OUT.filter.bad_marked = $("f_order").classList.contains("bad");
        $("clear").click(); OUT.filter.cleared = ST.sel; }
      // robot selection on the map, package selection in the table
      setTick(P.robot_tick); const fr = frame(tickNow()); mapPress([fr[0][0], fr[0][1]]);
      OUT.robot_select = {expected: R.m.robots[0].name, sel: ST.sel.robot};
      const prow = document.querySelector("#packages tr[data-pkg]");
      if (prow) { press(prow); OUT.package_select = {pkg: prow.dataset.pkg, sel: ST.sel.pkg, rows: $("seldetail").querySelectorAll("tr[data-t]").length}; }
      // decision row -> task
      if (P.decision_tick != null) { setTick(P.decision_tick); const dr = document.querySelector("#decision tr[data-coal]");
        if (dr) { press(dr); OUT.decision_select = {coal: dr.dataset.coal, sel: ST.sel}; } }
      // stations and docks
      OUT.resources = {stations: [...document.querySelectorAll("#resources tr[data-station]")].map(tr => [tr.dataset.station, tr.children[2].textContent, tr.children[3].textContent]),
                       docks: [...document.querySelectorAll("#resources tr[data-charger]")].map(tr => [tr.dataset.charger, tr.children[2].textContent])};
      if (R.chargers.length) { mapPress(R.chargers[0].cell); const a = JSON.parse(JSON.stringify(ST.sel)); mapPress(R.chargers[0].cell); OUT.dock_cycle = [a, JSON.parse(JSON.stringify(ST.sel))]; }
      // event row jumps
      const er = document.querySelector("#events tr[data-t]"); if (er) { const t = Number(er.dataset.t); press(er); OUT.event_jump = {to: t, now: tickNow()}; }
      // exports
      for (const opt of [...$("export").options].map(o => o.value).filter(Boolean)) { $("export").value = opt; $("export").onchange(); }
    } catch (e) { OUT.errors.push("harness: " + e + " " + (e.stack || "")); }
  }
  window.addEventListener("load", () => { run().then(() => { const p = document.createElement("pre"); p.id = "BT_RESULT";
    p.textContent = JSON.stringify(OUT); document.body.append(p); }); });
})();
