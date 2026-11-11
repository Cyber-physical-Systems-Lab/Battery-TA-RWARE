"""Model 2.2 behavioural gate (criteria fixed in GATE.md before the first run).

    python experiments/model_2_2/gate.py --jobs 30            # runs every episode, prints the verdict
    python experiments/model_2_2/gate.py --quick              # 1 seed per group (smoke)

Writes experiments/model_2_2/results/gate.json (git-ignored). Exit status 0 iff every criterion
passes. The per-tick monitor follows the model 2.1 gate: it wraps the PIBT step to see every
unit's goal and move, and checks occupancy, swaps and docks after every tick.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "docs"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "orders_v2"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "orders"))

STREAK, OSC, SEPARATE = 100, 10, 10


class Monitor:
    def __init__(self, sim):
        import battery_tarware.traffic as T
        self.sim, self.T = sim, T
        self.c = Counter()
        self.streak, self.osc, self.pairs = {}, {}, {}
        self.max_streak = self.max_osc = 0
        self.v = defaultdict(list)
        self.h1_run = {}
        if not hasattr(T.PIBT, "_gate_orig_step"):           # wrap once per process; dispatch to
            T.PIBT._gate_orig_step = T.PIBT.step              # the monitor of the current episode

            def step(pibt, units):
                T.PIBT._gate_orig_step(pibt, units)
                if T.PIBT._gate_monitor is not None:
                    T.PIBT._gate_monitor._movement(pibt, units)
            T.PIBT.step = step
        T.PIBT._gate_monitor = self

    def close(self):
        if getattr(self.T.PIBT, "_gate_monitor", None) is self:
            self.T.PIBT._gate_monitor = None

    def _work_cells(self):
        work = {}
        for u in self.sim.tasks.values():
            if not u.done and u.robots():
                if u.stage < len(u.stage_cells):
                    work.setdefault(u.stage_cells[u.stage], set()).add(u.id)
                if u.payload_cell is not None:
                    work.setdefault(u.payload_cell, set()).add(u.id)
        return work

    def _movement(self, pibt, units):
        sim = self.sim
        work = self._work_cells()
        working = {x.id for u in sim.tasks.values() for x in (sim._stage_crew(u) or [])}
        for u in units:                        # H1a: the goal movement actually used for a waiting helper
            r = sim.robots[u.id]
            if r.task_id is None or not r.operational or r.id in working or u.goal is None:
                continue
            t = sim.tasks[r.task_id]
            if r.role == t.type.carrier_role or sim._option(r, t).presence != "adjacent":
                continue
            if u.goal != t.stage_cell and work.get(u.goal, set()) - {t.id}:
                self.v["H1a"].append((sim.tick, r.id, u.goal))
        for u in units:
            moved = u.next is not None and u.next != u.cell
            o = self.osc.setdefault(u.id, [None, u.cell, 0])
            if moved:
                o[2] = o[2] + 1 if (o[0] == u.next and o[1] == u.cell) else 1
                o[0], o[1] = u.cell, u.next
                self.max_osc = max(self.max_osc, o[2])
                if o[2] == OSC:
                    self.v["L2"].append((self.sim.tick, u.id, u.cell))
            if u.goal is None or u.goal == u.cell:
                self.streak.pop(u.id, None)
                continue
            d = int(pibt.distances(u.zones, u.goal)[u.cell])
            s = self.streak.get(u.id)
            if s is None or s[0] != u.goal:
                self.streak[u.id] = [u.goal, d, 0]
            elif d < s[1]:
                s[1], s[2] = d, 0
            else:
                s[2] += 1
                self.max_streak = max(self.max_streak, s[2])
                if s[2] == STREAK:
                    self.v["L1"].append((self.sim.tick, u.id, u.cell, u.goal))

    def after_tick(self, prev):
        sim = self.sim
        by_cell = defaultdict(list)
        for r in sim.robots:
            by_cell[r.cell].append(r)
        pairs_now = {}
        for cell, rs in by_cell.items():
            if len(rs) < 2:
                continue
            tids = {r.task_id for r in rs}
            joint = len(tids) == 1 and None not in tids and all(r.role == sim.tasks[r.task_id].type.carrier_role for r in rs)
            t = sim.tasks.get(next(iter(tids))) if len(tids) == 1 and None not in tids else None
            crew = t is not None and cell in (t.stage_cell, t.payload_cell)
            key = frozenset(r.id for r in rs)
            if joint or crew:
                pairs_now[(key, cell)] = 0
                continue
            prevs = [n for (g, c), n in self.pairs.items() if c == cell and key <= g]
            if prevs and min(prevs) + 1 <= SEPARATE:
                pairs_now[(key, cell)] = min(prevs) + 1
                continue
            self.v["S1"].append((sim.tick, cell, sorted(key)))
        self.pairs = pairs_now
        cur = {r.id: r.cell for r in sim.robots}
        at_prev = {c: i for i, c in prev.items()}
        for i, c in cur.items():
            j = at_prev.get(c)
            if j is not None and j != i and cur[j] == prev[i] and c != prev[i] and i < j:
                same = sim.robots[i].task_id is not None and sim.robots[i].task_id == sim.robots[j].task_id
                if not same:
                    self.v["S2"].append((sim.tick, i, j))
        holders, charging = defaultdict(list), defaultdict(list)
        for r in sim.robots:
            if r.charger is not None:
                holders[r.charger].append(r.id)
            if r.charging:
                charging[r.charger].append(r.id)
                if r.cell != sim.chargers[r.charger].cell:
                    self.v["S3"].append((sim.tick, "charging_off_dock", r.id))
        for cid, ids in list(holders.items()) + list(charging.items()):
            if len(ids) > 1:
                self.v["S3"].append((sim.tick, "dock_shared", cid, ids))
        for c in sim.chargers:
            rb = getattr(c, "reserved_by", None)
            if rb is not None and sim.robots[rb].charger != c.id:
                self.v["S3"].append((sim.tick, "stale_reservation", c.id, rb))
        # H1b: a waiting helper is not held on a foreign work cell (also measured under 2.1)
        work = self._work_cells()
        working = {x.id for u in sim.tasks.values() for x in (sim._stage_crew(u) or [])}
        on_foreign = set()
        for r in sim.robots:
            if r.task_id is None or not r.operational or r.id in working:
                continue
            t = sim.tasks[r.task_id]
            if r.role == t.type.carrier_role or sim._option(r, t).presence != "adjacent":
                continue
            foreign = {c for c, ids in work.items() if ids - {t.id}} - {t.stage_cell}
            if r.cell in foreign:
                on_foreign.add(r.id)
                self.h1_run[r.id] = self.h1_run.get(r.id, 0) + 1
                if self.h1_run[r.id] == SEPARATE + 1:
                    self.v["H1b"].append((sim.tick, r.id, r.cell))   # stays on one too long
        for i in list(self.h1_run):
            if i not in on_foreign:
                del self.h1_run[i]


def _episode(job):
    import battery_tarware as bt
    kind, name, model, rule, seed, ticks = job["kind"], job["name"], job["model"], job["rule"], job["seed"], job["ticks"]
    try:
        if kind == "builtin":
            cfg = copy.deepcopy(bt.get_scenario(name, model=model))
        elif kind == "large":
            import make_gifs
            cfg = copy.deepcopy(make_gifs.large_scenarios(model=model)[name])
        else:
            import workloads
            cfg = workloads.build(name, model, "standard", max_steps=ticks)
        kw = {"action_mode": "primitive"} if job.get("primitive") else {}
        env = bt.make(cfg, **kw)
        sim = env.unwrapped.sim
        mon = Monitor(sim)
        if kind == "orders":
            from battery_tarware.orders.scheduler import OrderScheduler
            co = OrderScheduler(rule)
        elif job.get("primitive"):
            co = bt.RandomCoordinator(seed=seed)
        else:
            co = bt.GreedyCoalitionCoordinator("fifo")
        env.reset(seed=seed)
        co.reset(env)
        while True:
            prev = {r.id: r.cell for r in sim.robots}
            _, _, term, trunc, _ = env.step(co.act(env))
            mon.after_tick(prev)
            if term or trunc or (ticks and sim.tick >= ticks):
                break
        mon.close()
        ev = sim.events.events
        m = sim.metrics()
        return {**{k: v for k, v in job.items()}, "ticks_run": sim.tick, "n": {k: len(v) for k, v in mon.v.items()},
                "examples": {k: v[:3] for k, v in mon.v.items()}, "max_no_progress": mon.max_streak,
                "max_osc": mon.max_osc, "stalls": sum(1 for e in ev if e.type == "stall_suspected"),
                "stranded": int(m.get("stranded", 0)), "tasks_completed": int(m.get("tasks_completed", 0)),
                "orders_completed": sum(1 for e in ev if e.type == "order_completed")}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {**job, "error": "%s: %s" % (type(e).__name__, e), "tb": traceback.format_exc()[-800:]}


def jobs(quick=False):
    seeds5, seeds2, seeds10 = ((0,), (0,), (0,)) if quick else (range(5), range(2), range(10))
    out = []
    for model in ("2.1", "2.2"):
        for name in ("basic", "intermediate", "advanced"):
            out += [{"kind": "builtin", "name": name, "model": model, "rule": "fifo", "seed": s, "ticks": None} for s in seeds5]
            out += [{"kind": "large", "name": name, "model": model, "rule": "fifo", "seed": s, "ticks": None} for s in seeds2]
            out += [{"kind": "orders", "name": name, "model": model, "rule": r, "seed": s, "ticks": 3000}
                    for r in ("fifo", "nsp", "plb") for s in seeds10]
        out += [{"kind": "orders_factors", "name": "base", "model": model, "rule": "plb", "seed": 1017, "ticks": 600},
                {"kind": "orders_factors", "name": "rate=0.05", "model": model, "rule": "plb", "seed": 6, "ticks": 3000}]
    out += [{"kind": "builtin", "name": n, "model": "2.2", "rule": "random", "seed": 0, "ticks": 500, "primitive": True}
            for n in ("basic", "intermediate", "advanced")]
    return out


def _factors_episode(job):
    import factors
    from battery_tarware.orders.run import run_episode
    cfg = factors.build(**({"rate": 0.05} if job["name"] == "rate=0.05" else {}))
    cfg["model"] = job["model"]
    res = run_episode(cfg, job["rule"], job["seed"], job["ticks"], 0, record=False)
    return {**job, "stalls": sum(1 for e in res["sim"].events.events if e.type == "stall_suspected"),
            "orders_completed": res["metrics"]["order"]["completed"], "n": {}}


def run(job):
    return _factors_episode(job) if job["kind"] == "orders_factors" else _episode(job)


def verdict(rows):
    v22 = [r for r in rows if r["model"] == "2.2" and not r.get("error")]
    by21 = {(r["kind"], r["name"], r["rule"], r["seed"]): r for r in rows if r["model"] == "2.1" and not r.get("error")}
    tot = lambda k, rs: sum(r.get("n", {}).get(k, 0) for r in rs)
    res = {}
    for k in ("S1", "S2", "S3", "H1a", "H1b", "L2"):
        res[k] = (tot(k, v22) == 0, "%d violations in %d episodes" % (tot(k, v22), len(v22)))
    bi = [r for r in v22 if r["kind"] in ("builtin", "large") and not r.get("primitive")]
    res["L1"] = (tot("L1", bi) == 0, "%d no-progress streaks >= %d ticks in %d task-only episodes" % (tot("L1", bi), STREAK, len(bi)))
    worse = [r for r in v22 if r["kind"] != "orders_factors" and (r["kind"], r["name"], r["rule"], r["seed"]) in by21
             and r["stranded"] > by21[(r["kind"], r["name"], r["rule"], r["seed"])]["stranded"]]
    res["L3"] = (not worse, "%d episodes strand more robots than under 2.1" % len(worse))
    od = [r for r in v22 if r["kind"] == "orders"]
    res["D1"] = (sum(r["stalls"] for r in od) == 0 and len(od) > 0, "%d suspected stalls in %d order episodes" % (sum(r["stalls"] for r in od), len(od)))
    kg = [r for r in rows if r["kind"] == "orders_factors"]
    ok = all(r["stalls"] == 0 for r in kg if r["model"] == "2.2") and all(r["stalls"] > 0 for r in kg if r["model"] == "2.1")
    res["D2"] = (ok and len(kg) == 4, "; ".join("%s %s seed %d: %d stalls" % (r["model"], r["name"], r["seed"], r["stalls"]) for r in kg))
    errs = [r for r in rows if r.get("error")]
    res["errors"] = (not errs, "%d episodes raised" % len(errs))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "gate.json"))
    a = ap.parse_args(argv)
    js = jobs(a.quick)
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(run, js, chunksize=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, default=str)
    res = verdict(rows)
    print("=== model 2.2 gate (%d episodes) ===" % len(rows))
    for k, (ok, msg) in res.items():
        print("  %-6s %s  %s" % (k, "PASS" if ok else "FAIL", msg))
    print("\n  contrast, model 2.1: H1a %d, H1b %d, L1 (task-only) %d, order stalls %d" % (
        sum(r.get("n", {}).get("H1a", 0) for r in rows if r["model"] == "2.1"),
        sum(r.get("n", {}).get("H1b", 0) for r in rows if r["model"] == "2.1"),
        sum(r.get("n", {}).get("L1", 0) for r in rows if r["model"] == "2.1" and r["kind"] in ("builtin", "large")),
        sum(r.get("stalls", 0) for r in rows if r["model"] == "2.1" and r["kind"] == "orders")))
    for r in rows:
        if r.get("error"):
            print("ERROR", r["kind"], r["name"], r["model"], r["rule"], r["seed"], r["error"], r.get("tb", "")[-300:])
    return 0 if all(ok for ok, _ in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
