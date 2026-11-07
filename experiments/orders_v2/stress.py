"""Long-horizon stress runs (18,000 ticks = 25 h of simulated time at 5 s/tick) with system checks.

    python experiments/orders_v2/stress.py [--ticks 18000] [--jobs 30] [--rules fifo,nsp,plb] [--models 2.1,2.2]

Stress cases (experiments/orders_v2/workloads.py):
  high_demand          basic, standard profile, 0.13 orders/tick (above capacity: sustained backlog)
  battery_scarcity     basic, batteries x0.12, half the docks
  charging_congestion  intermediate, batteries x0.12, half the fast/slow docks (compatibility limits)
  dense_traffic        basic map with the fleet doubled (12 AGVs + 6 picking robots), 0.13 orders/tick
  complex_dependencies advanced, dag profile (2-4 lines, 70% chained, half compound)
  sustained_backlog    intermediate, standard profile, 0.09 orders/tick (above capacity)

Checks per run (each run in its own process): no exception (the order layer checks the inventory
invariants every tick and raises on a violation); inventory rebuilt from the event log alone equals
the final inventory for every package; package conservation; every order-task transition allowed
by the lifecycle; event ticks non-decreasing; every completed order has only completed tasks and
closes after its last task; manager counts equal the counts from the event log. Reported: suspected
stalls (deadlock), starving tasks and the longest wait for a first assignment, unfinished orders
and their age, backlog slope, peak memory (RSS) of the run's process, trace and dashboard file
sizes, scheduler P99 and deadline violations. Writes experiments/orders_v2/results/stress/.
"""
import argparse
import json
import os
import resource
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results", "stress")

CASES = {
    "high_demand": dict(base="basic", profile="standard", rate=0.13),
    "battery_scarcity": dict(base="basic", profile="standard", battery="tight"),
    "charging_congestion": dict(base="intermediate", profile="standard", battery="tight"),
    "dense_traffic": dict(base="basic", profile="standard", rate=0.13, fleet_x=2),
    "complex_dependencies": dict(base="advanced", profile="dag"),
    "sustained_backlog": dict(base="intermediate", profile="standard", rate=0.09),
}


def scenario(case, model, ticks):
    import workloads
    c = dict(CASES[case])
    fx = c.pop("fleet_x", 1)
    cfg = workloads.build(c.pop("base"), model, c.pop("profile"), max_steps=ticks, **c)
    if fx > 1:
        for g in cfg["fleet"]:
            g["count"] = g["count"] * fx
    cfg["name"] = "stress_%s_%s" % (case, model)
    return cfg


def checks(sim, trace):
    from battery_tarware.orders.lifecycle import TRANSITIONS, OTaskState
    from battery_tarware.orders.reports import replay_inventory
    bad = []
    mgr, inv, ev = sim.orders, sim.orders.inventory, trace["events"]
    rp = replay_inventory(ev)
    for pid, p in inv.packages.items():
        r = rp.get(pid)
        if r is None or (r["status"], r["shelf"], r["loc"], r["reserved"]) != (p.status, p.shelf_id, inv.rack_location_of(p), p.reserved_by):
            bad.append("replay differs for %s" % pid)
            break
    n_arrived = sum(1 for e in ev if e[1] == "inventory" and e[4]["op"] == "package_arrived")
    n_init = sum(len(v[1]) for e in ev if e[1] == "inventory" and e[4]["op"] == "initialised" for v in e[4]["shelves"].values())
    if len(inv.packages) != n_init + n_arrived:
        bad.append("package conservation: %d packages, %d initial + %d arrived" % (len(inv.packages), n_init, n_arrived))
    last = -1
    state = {}
    for e in ev:
        if e[0] < last:
            bad.append("event ticks decrease at %d" % e[0])
            break
        last = e[0]
        if e[1] == "order_task":
            d = e[4]
            if d["frm"] is not None and OTaskState(d["to"]) not in TRANSITIONS[OTaskState(d["frm"])] and d["frm"] != d["to"]:
                bad.append("transition %s -> %s for %s" % (d["frm"], d["to"], d["task_id"]))
            if d["frm"] is not None and state.get(d["task_id"]) != d["frm"]:
                bad.append("transition of %s from %s, but log says %s" % (d["task_id"], d["frm"], state.get(d["task_id"])))
            state[d["task_id"]] = d["to"]
    done_t = {e[4]["task_id"]: e[0] for e in ev if e[1] == "order_task" and e[4]["to"] == "completed"}
    for o in mgr.orders.values():
        if o.state == "completed":
            ts = [mgr.tasks[i] for i in o.task_ids]
            if any(t.state.value != "completed" for t in ts) or o.closed < max(done_t.get(t.task_id, 0) for t in ts):
                bad.append("order %s completed with open tasks" % o.order_id)
                break
    for tid, t in mgr.tasks.items():
        if state.get(tid) != t.state.value:
            bad.append("task %s: manager %s, log %s" % (tid, t.state.value, state.get(tid)))
            break
    n_completed = sum(1 for e in ev if e[1] == "order_completed")
    if n_completed != sum(1 for o in mgr.orders.values() if o.state == "completed"):
        bad.append("completed orders: log %d, manager %d" % (n_completed, sum(1 for o in mgr.orders.values() if o.state == "completed")))
    return bad


def one(job):
    case, model, rule, seed, ticks = job
    t0 = time.time()
    try:
        from battery_tarware.orders.dashboard import write_dashboard
        from battery_tarware.orders.run import run_episode
        from battery_tarware.orders.trace import _json_default
        res = run_episode(scenario(case, model, ticks), rule, seed, ticks, 1000)
        sim, tr, m = res["sim"], res["trace"], res["metrics"]
        bad = checks(sim, tr)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.json")
            with open(p, "w") as f:
                json.dump(tr, f, default=_json_default, separators=(",", ":"))
            trace_mb = os.path.getsize(p) / 1e6
            t1 = time.time()
            h = write_dashboard([tr], os.path.join(d, "d.html"))
            dash_mb, dash_s = os.path.getsize(h) / 1e6, time.time() - t1
            if job == KEEP:
                os.makedirs(OUT, exist_ok=True)
                os.replace(p, os.path.join(OUT, "trace_%s_%s_%s_%d.json" % job[:4]))
        o, s = m["order"], m["scheduling"]
        lat = s["assignment_latency"]
        return {"job": job, "error": None, "checks_failed": bad, "ticks": sim.tick, "robots": m["meta"]["robots"],
                "orders_arrived": o["orders_arrived"], "completed": o["completed"], "failed": o["failed"],
                "cancelled": o["cancelled"], "unfinished": o["unfinished"], "unfinished_age_mean": o["unfinished_age_mean"],
                "unfinished_age_max": max([sim.tick - x.arrival for x in sim.orders.orders.values() if x.state == "open"] or [0]),
                "backlog_slope": o["backlog_slope_per_1000"], "throughput": o["throughput_per_1000"],
                "stalls": s["stalls_suspected"], "starving": s["starving_tasks"], "latency_max": lat["max"], "latency_p95": lat["p95"],
                "retries": s["retries"], "stranded": m["robot"]["stranded"], "charge_sessions": m["robot"]["charge_sessions"],
                "dock_queue_ticks": m["robot"]["dock_queue_ticks"], "min_soc": m["robot"]["min_soc"],
                "rt_p99": s["runtime_ms"]["p99"], "rt_max": s["runtime_ms"]["max"], "viol": s.get("deadline_violation_rate"),
                "events": len(tr["events"]), "trace_mb": trace_mb, "dashboard_mb": dash_mb, "dashboard_write_s": dash_s,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "wall_s": time.time() - t0,
                "unsuccessful": m["bottlenecks"]["unsuccessful_by_category"]}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"job": job, "error": "%s: %s" % (type(e).__name__, e), "tb": traceback.format_exc()[-1500:],
                "wall_s": time.time() - t0}


KEEP = ("complex_dependencies", "2.2", "nsp", 0, 18000)      # one long trace kept for the browser check


def main(argv=None):
    global KEEP
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=18000)
    ap.add_argument("--jobs", type=int, default=30)
    ap.add_argument("--rules", default="fifo,nsp,plb")
    ap.add_argument("--models", default="2.1,2.2")
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args(argv)
    KEEP = ("complex_dependencies", "2.2", "nsp", 0, a.ticks)
    jobs = [(c, mdl, r, s, a.ticks) for c in CASES for mdl in a.models.split(",") for r in a.rules.split(",")
            for s in range(a.seeds)]
    with ProcessPoolExecutor(a.jobs, max_tasks_per_child=1) as ex:
        rows = list(ex.map(one, jobs, chunksize=1))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "runs.json"), "w") as f:
        json.dump(rows, f, default=str)
    L = ["# Long-horizon stress runs (%d ticks)" % a.ticks, "",
         "Throughput per 1000 ticks after a 1000-tick warm-up; age in ticks; memory = peak RSS of the run's process "
         "(simulation + trace recording + metrics + dashboard); P99 = scheduler ms per tick; viol. = share of ticks over 20 ms.", "",
         "| case | model | rule | robots | arrived | completed | unfinished (max age) | backlog slope | stalls | starving (max wait) | stranded | checks | peak MB | trace MB | dashboard MB | P99 ms | viol. % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c, mdl, rule, seed, _ = r["job"]
        if r.get("error"):
            L.append("| %s | %s | %s | | | | | | | | | ERROR: %s | | | | | |" % (c, mdl, rule, r["error"][:120]))
            continue
        L.append("| %s | %s | %s | %d | %d | %d | %d (%d) | %+.1f | %d | %d (%s) | %d | %s | %.0f | %.1f | %.1f | %.1f | %.2f |" % (
            c, mdl, rule, r["robots"], r["orders_arrived"], r["completed"], r["unfinished"], r["unfinished_age_max"],
            r["backlog_slope"] or 0, r["stalls"], r["starving"], "%.0f" % r["latency_max"] if r["latency_max"] is not None else "–",
            r["stranded"], "PASS" if not r["checks_failed"] else "FAIL: " + "; ".join(r["checks_failed"][:2]),
            r["peak_rss_mb"], r["trace_mb"], r["dashboard_mb"], r["rt_p99"], 100 * (r["viol"] or 0)))
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(os.path.join(OUT, "summary.md"))


if __name__ == "__main__":
    main()
