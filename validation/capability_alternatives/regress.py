"""Short behavioural regression for role alternatives (intermediate, substitution=True).

  python validation/capability_alternatives/regress.py [seeds] [procs]

Runs the greedy baseline on the substitution variant of intermediate (standard and large map, inline
docks and dock bays) and applies the model 2.1 gate's per-tick monitor (validation/model_2_1/gate.py:
S1p co-location, S2 swaps, S3 dock exclusivity, L1 no-progress >= 100 ticks, L2 oscillation >= 10),
plus checks for the new semantics:

  X1  a robot filling a role through an ``adjacent`` alternative never works on the stage cell;
  X2  a robot filling a role through an ``adjacent`` alternative works within its radius;
  X3  no robot holds two roles of one task (carrier and support are distinct robots);
  X4  no stranding.

It also reports how often the tote AGVs took the support role. Thresholds are the gate's (frozen).
"""
from __future__ import annotations

import copy
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs"))
sys.path.insert(0, os.path.join(ROOT, "validation", "model_2_1"))

import battery_tarware as bt        # noqa: E402
import gate                         # noqa: E402
import make_gifs                    # noqa: E402

VARIANTS = ("intermediate_sub", "intermediate_sub_large")
LAYOUTS = ("inline", "bays")


def scenario(name, layout):
    cfg = bt.get_scenario("intermediate", substitution=True)
    if name.endswith("_large"):
        cfg = make_gifs.scale(cfg, 2, generator=dict(make_gifs.LARGE_RACKS, goals=7, docks=12))
    cfg = copy.deepcopy(cfg)
    cfg["map"] = dict(cfg["map"], dock_layout=layout)
    return cfg


def episode(args):
    name, layout, seed = args
    env = bt.make(scenario(name, layout))
    sim = env.unwrapped.sim
    mon = gate.Monitor(sim)
    co = bt.GreedyCoalitionCoordinator("fifo")
    env.reset(seed=seed)
    co.reset(env)
    x = {"X1": 0, "X2": 0, "X3": 0}
    support = {"agv": 0, "picker": 0}
    seen = set()
    while True:
        prev = {r.id: r.cell for r in sim.robots}
        _, _, te, tr, info = env.step(co.act(env))
        mon.after_tick(prev)
        for t in sim.tasks.values():
            ids = t.robots()
            if len(ids) != len(set(ids)):
                x["X3"] += 1
        for r in sim.robots:
            if r.task_id is None or not r.worked:
                continue
            t = sim.tasks[r.task_id]
            rs = t.type.role(r.role)
            if not rs.alternatives:
                continue
            opt = rs.option_for(r)
            if opt.presence == "adjacent":
                if r.cell == t.stage_cell:
                    x["X1"] += 1
                if sim.chebyshev(r.cell, t.stage_cell) > opt.radius:
                    x["X2"] += 1
            key = (t.id, t.stage, r.id)
            if key not in seen:
                seen.add(key)
                support["agv" if r.can("load_assist") else "picker"] += 1
        if te or tr:
            break
    m = info["metrics"]
    return {"scenario": name, "layout": layout, "seed": seed, "ticks": sim.tick,
            "tasks": m["tasks_completed"], "stranded": m["stranded"],
            "n_violations": {k: len(v) for k, v in mon.violations.items()},
            "max_no_progress": mon.max_streak, "max_oscillation": mon.max_osc,
            "X": x, "support_stages": support}


if __name__ == "__main__":
    seeds = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0,1,2,3,4,5,6,7,8,9").split(",")]
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = [(n, l, s) for n in VARIANTS for l in LAYOUTS for s in seeds]
    with Pool(procs) as p:
        res = p.map(episode, jobs)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "out", "regress.json"), "w"))
    ok = True
    for n in VARIANTS:
        for l in LAYOUTS:
            rs = [r for r in res if r["scenario"] == n and r["layout"] == l]
            v = {}
            for r in rs:
                for k, c in r["n_violations"].items():
                    v[k] = v.get(k, 0) + c
            xs = {k: sum(r["X"][k] for r in rs) for k in ("X1", "X2", "X3")}
            strand = sum(r["stranded"] for r in rs)
            npmax = max(r["max_no_progress"] for r in rs)
            osc = max(r["max_oscillation"] for r in rs)
            agv = sum(r["support_stages"]["agv"] for r in rs)
            pk = sum(r["support_stages"]["picker"] for r in rs)
            bad = {k: c for k, c in v.items() if k != "S1" and c} or {}
            passed = not bad and not any(xs.values()) and strand == 0 and npmax < gate.STREAK and osc < gate.OSC
            ok &= passed
            print("%-24s %-6s %s  tasks %.1f  violations %s (S1 registered rule, reported only: %d)  X %s  stranded %d  "
                  "max no-progress %d  max oscillation %d  support stages: AGV %d / picker %d"
                  % (n, l, "PASS" if passed else "FAIL", sum(r["tasks"] for r in rs) / len(rs), bad or "none",
                     v.get("S1", 0), xs, strand, npmax, osc, agv, pk))
    print("OVERALL", "PASS" if ok else "FAIL")
