"""Automated browser checks of the replay dashboard against the authoritative trace.

    python experiments/checks/dashboard_browser.py trace.json [--ticks 12]

Builds the dashboard for one trace, opens it in headless Chrome/Chromium with a test plan
(dashboard_harness.js), drives playback, kanban, filters, dependency graph, robot / package /
decision / dock selection, exports and a deep link through the page's own handlers, and compares
what the page shows with what the event log and frames say. Needs Chrome or Chromium on PATH (or
$BT_CHROME); prints the failed checks and exits 1 if any. Used by tests/test_dashboard_browser.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
STATES = ["pending", "blocked", "ready", "assigned", "executing", "suspended", "retrying", "failed", "completed", "cancelled"]


def find_chrome() -> Optional[str]:
    for c in (os.environ.get("BT_CHROME"), "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome"):
        if c and shutil.which(c):
            return shutil.which(c)
    return None


def run_page(html: str, plan: Dict, hash_: str = "", budget_ms: int = 20000, timeout: int = 240) -> Dict:
    """Open ``html`` (dashboard file) with the harness and ``plan``; return the observations."""
    chrome = find_chrome()
    if chrome is None:
        raise RuntimeError("no Chrome/Chromium found (set BT_CHROME)")
    src = open(html).read()
    inject = "<script>window.__BT_PLAN = %s;</script><script>%s</script>" % (
        json.dumps(plan), open(os.path.join(HERE, "dashboard_harness.js")).read())
    i = src.rindex("</body>")
    page = src[:i] + inject + src[i:]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "page.html")
        with open(p, "w") as f:
            f.write(page)
        cmd = [chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--window-size=1400,1000",
               "--virtual-time-budget=%d" % budget_ms, "--user-data-dir=%s" % os.path.join(d, "prof"),
               "--dump-dom", "file://%s%s" % (p, hash_)]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    i = out.rfind('<pre id="BT_RESULT">')
    if i < 0:
        raise RuntimeError("harness did not report (page error or timeout)")
    raw = out[i + len('<pre id="BT_RESULT">'):out.index("</pre>", i)]
    import html as H
    return json.loads(H.unescape(raw))


# ───────────────────────────── expectations from the trace
def task_states(trace: Dict, t: int) -> Dict[str, str]:
    st = {}
    for tick, typ, _, _, d in trace["events"]:
        if tick > t:
            break
        if typ == "order_task":
            st[d["task_id"]] = d["to"]
    return st


def open_orders(trace: Dict, t: int) -> int:
    n = 0
    for o in trace["tables"]["orders"]:
        if o["arrival"] <= t and (o["closed"] is None or o["closed"] > t):
            n += 1
    return n


def plan_for(trace: Dict, n_ticks: int = 12) -> Dict:
    meta = trace["meta"]
    T0, N = meta["first_tick"], meta["ticks"]
    ticks = sorted({T0 + int(k * (N - 1) / max(1, n_ticks - 1)) for k in range(n_ticks)})
    dec = sorted(e[0] for e in trace["events"] if e[1] == "schedule_decision" and e[4]["tasks"])
    multi = [o for o in trace["tables"]["orders"] if "|" in (o["tasks"] or "")]
    deps = [o for o in multi if any(r["depends_on"] for r in trace["tables"]["tasks"] if r["order_id"] == o["order_id"])]
    pick = (deps or multi or trace["tables"]["orders"])[0]
    mid = T0 + N // 2
    opn = [o for o in trace["tables"]["orders"] if o["arrival"] <= mid and (o["closed"] is None or o["closed"] > mid)]
    fo = (opn or trace["tables"]["orders"])[0]
    return {"ticks": ticks + ([dec[len(dec) // 2]] if dec else []), "play_from": T0 + N // 3,
            "dag_order": pick["order_id"], "dag_tick": min(T0 + N - 1, (pick["closed"] or pick["arrival"]) - 1 if pick["closed"] else pick["arrival"] + 5),
            "filter_order": {"short": str(int(fo["order_id"][1:])), "id": fo["order_id"], "tick": mid},
            "robot_tick": mid, "decision_tick": dec[len(dec) // 2] if dec else None}


def check(trace: Dict, obs: Dict, plan: Dict) -> List[str]:
    """Failed checks (empty = all agree)."""
    bad: List[str] = []
    meta = trace["meta"]
    T0 = meta["first_tick"]
    if obs["errors"]:
        bad.append("page errors: %s" % obs["errors"][:3])
    acts = meta["activities"]
    for t_s, k in obs["kanban"].items():
        t = int(t_s)
        exp = task_states(trace, t)
        cnt = {s: 0 for s in STATES}
        for s in exp.values():
            cnt[s] += 1
        if k["counts"] != cnt:
            bad.append("kanban counts at %d: page %s, log %s" % (t, k["counts"], cnt))
        wrong = [(tid, s, exp.get(tid)) for tid, s in k["cards"].items() if exp.get(tid) != s]
        if wrong:
            bad.append("kanban cards at %d in the wrong column: %s" % (t, wrong[:3]))
        if obs["orders_open"][t_s] != open_orders(trace, t):
            bad.append("open orders at %d: page %d, log %d" % (t, obs["orders_open"][t_s], open_orders(trace, t)))
        fr = trace["frames"][t - T0]
        for i, (name, act, batt) in enumerate(obs["robots"][t_s]):
            pct = math.floor(100 * fr[i][2] + 0.5)               # JavaScript Math.round
            if act != acts[fr[i][6]] or batt != "%d%%" % pct:
                bad.append("robot %s at %d: page %s %s, frame %s %d%%" % (name, t, act, batt, acts[fr[i][6]], pct))
                break
        d = [e for e in trace["events"] if e[0] == t and e[1] == "schedule_decision"]
        if d and [[r[0], r[5]] for r in d[0][4]["tasks"]] != obs["decisions"][t_s]:
            bad.append("decision table at %d differs from the log" % t)
    pb = obs.get("playback") or {}
    if not pb or pb["to"] <= pb["from"]:
        bad.append("playback did not advance: %s" % pb)
    ps = obs.get("play_select")
    if not ps or ps["sel"] != ps["task"] or ps["task"] not in (ps["head"] or ""):
        bad.append("selecting a card during playback failed: %s" % ps)
    dag = obs.get("dag")
    if plan.get("dag_order"):
        rows = [r for r in trace["tables"]["tasks"] if r["order_id"] == plan["dag_order"]]
        edges = sum(len([x for x in (r["depends_on"] or "").split("|") if x]) for r in rows)
        if not dag or sorted(dag["nodes"]) != sorted(r["task_id"] for r in rows) or dag["edges"] != edges:
            bad.append("dependency graph of %s: page %s, log %d tasks %d edges" % (plan["dag_order"], dag, len(rows), edges))
        else:
            exp = task_states(trace, plan["dag_tick"])
            for tid, st in dag["rows"]:
                if exp.get(tid, "") != st and not (tid not in exp and st in ("–", "–")):
                    bad.append("order task %s state at %d: page %s, log %s" % (tid, plan["dag_tick"], st, exp.get(tid)))
            if not set(dag["highlighted"]) <= {r["task_id"] for r in rows}:
                bad.append("kanban highlights tasks of other orders: %s" % dag["highlighted"])
            if dag["node_select"].get("task") not in dag["nodes"]:
                bad.append("dependency-graph node did not select its task")
    f = obs.get("filter")
    if plan.get("filter_order"):
        fid = plan["filter_order"]["id"]
        if not f or f["sel"].get("order") != fid or f["bad"] or any(o != fid for o in f["cards"]) \
                or any(o != fid for o in f["orders_rows"]) or not f["bad_marked"] or f["cleared"]:
            bad.append("order filter %r: %s" % (plan["filter_order"]["short"], f))
    rs = obs.get("robot_select") or {}
    if rs.get("sel") != rs.get("expected"):
        bad.append("robot selection on the map: %s" % rs)
    pk = obs.get("package_select")
    if pk and (pk["sel"] != pk["pkg"] or pk["rows"] < 1):
        bad.append("package selection: %s" % pk)
    ds = obs.get("decision_select")
    if plan.get("decision_tick") is not None and (not ds or not (ds["sel"].get("task") or ds["sel"].get("coal"))):
        bad.append("decision row selection: %s" % ds)
    res = (trace.get("metrics") or {}).get("resources")
    if res:
        exp_st = {"%d,%d" % tuple(s["cell"]): [str(s["shipped"]), str(s["received"])] for s in res["stations"]}
        for key, sh, rc in obs["resources"]["stations"]:
            if exp_st.get(key) != [sh, rc]:
                bad.append("station %s: page %s/%s, metrics %s" % (key, sh, rc, exp_st.get(key)))
        exp_dk = {str(c["charger"]): str(c["sessions"]) for c in res["chargers"]}
        for key, n in obs["resources"]["docks"]:
            if exp_dk.get(key) != n:
                bad.append("dock %s sessions: page %s, metrics %s" % (key, n, exp_dk.get(key)))
    dc = obs.get("dock_cycle")
    if meta.get("chargers") and dc is not None and not any("charger" in s for s in dc):
        bad.append("dock selection on the map: %s" % dc)
    ej = obs.get("event_jump")
    if ej and ej["to"] != ej["now"]:
        bad.append("event row jump: %s" % ej)
    kinds = ("orders.csv", "tasks.csv", "packages.csv", "shelves.csv", "robots.csv", "trips.csv", "stations.csv",
             "chargers.csv", "events.csv", "metrics.json", "trace.json")
    names = {k: d for d in obs["downloads"] for k in kinds if d["name"].endswith("_" + k)}
    tables = trace.get("tables") or {}
    for k in ("orders", "tasks", "packages", "shelves", "robots", "trips"):
        d = names.get(k + ".csv")
        if d is None or (tables.get(k) and d["lines"] != len(tables[k]) + 1):
            bad.append("export %s.csv: %s lines, table has %d rows" % (k, d and d["lines"], len(tables.get(k) or [])))
    if "events.csv" not in names or names["events.csv"]["lines"] != len(trace["events"]) + 1:
        bad.append("export events.csv: %s lines, log has %d events" % (names.get("events.csv", {}).get("lines"), len(trace["events"])))
    for k in ("metrics.json", "trace.json", "stations.csv", "chargers.csv"):
        if k not in names:
            bad.append("export %s missing" % k)
    return bad


def check_deep_link(html: str, trace: Dict) -> List[str]:
    meta = trace["meta"]
    t = meta["first_tick"] + meta["ticks"] // 2
    oid = trace["tables"]["orders"][0]["order_id"]
    obs = run_page(html, {"deep_link_only": True}, hash_="#run=0&t=%d&order=%s" % (t, oid), budget_ms=5000)
    bad = []
    if obs["errors"]:
        bad.append("deep link page errors: %s" % obs["errors"][:3])
    if obs["initial_tick"] != t or obs["initial_sel"].get("order") != oid:
        bad.append("deep link #t=%d&order=%s opened tick %s, selection %s" % (t, oid, obs["initial_tick"], obs["initial_sel"]))
    if meta.get("chargers"):
        obs = run_page(html, {"deep_link_only": True}, hash_="#t=%d&charger=0" % t, budget_ms=5000)
        if obs["initial_sel"].get("charger") != "0":
            bad.append("deep link charger=0: selection %s" % obs["initial_sel"])
    return bad


def verify(trace: Dict, n_ticks: int = 12, html: Optional[str] = None) -> List[str]:
    from battery_tarware.orders.dashboard import write_dashboard
    with tempfile.TemporaryDirectory() as d:
        html = html or os.path.join(d, "dashboard.html")
        write_dashboard([trace], html)
        plan = plan_for(trace, n_ticks)
        obs = run_page(html, plan)
        return check(trace, obs, plan) + check_deep_link(html, trace)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--ticks", type=int, default=12)
    a = ap.parse_args(argv)
    with open(a.trace) as f:
        trace = json.load(f)
    bad = verify(trace, a.ticks)
    for b in bad:
        print("FAIL", b)
    print("dashboard agrees with the trace" if not bad else "%d checks failed" % len(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
