"""Paired-seed decomposition of a throughput difference between two gate configurations (descriptive).

  python validation/model_2_1/throughput.py <scenario> <config_a> <config_b> <seeds> [procs]

For every completed task: release -> first assignment (queue wait), first assignment -> first stage
started (crew gathering and travel), first stage -> completion (execution incl. transport and
synchronized stages). Per episode: robot-ticks by activity (moving, stalled with a target, waiting at a
target for the crew, working, charging, heading to a dock, idle), deliveries, energy. Differences are
paired by seed; 95% CIs by a paired bootstrap (10,000 resamples, seed 0).
"""
import collections
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate                                         # noqa: E402
import battery_tarware as bt                        # noqa: E402
from battery_tarware.robots import RobotStatus      # noqa: E402
from battery_tarware.tasks.lifecycle import TaskState  # noqa: E402


def episode(args):
    sc, cf, seed = args
    env = bt.make(gate.scenario(sc, cf))
    sim = env.unwrapped.sim
    co = bt.GreedyCoalitionCoordinator("fifo")
    env.reset(seed=seed)
    rt = collections.Counter()
    while True:
        prev = {r.id: r.cell for r in sim.robots}
        _, _, te, tr, info = env.step(co.act(env))
        for r in sim.robots:
            if r.status == RobotStatus.CHARGE:
                rt["charging" if r.charging else "to_dock"] += 1
            elif r.status in (RobotStatus.FAILED, RobotStatus.STRANDED):
                rt["failed"] += 1
            elif r.status == RobotStatus.IDLE:
                rt["idle"] += 1
            elif r.worked:
                rt["working"] += 1
            elif r.cell != prev[r.id]:
                rt["task_moving"] += 1
            else:
                tgt = sim._target(r)
                rt["task_waiting_in_place" if tgt in (None, r.cell) else "task_blocked"] += 1
        if te or tr:
            break
    phases = collections.defaultdict(list)
    by_type = collections.Counter()
    for t in sim.tasks.values():
        if t.state != TaskState.COMPLETED:
            continue
        h = t.record.history
        assigned = next((k for k, a, b, _ in h if b == TaskState.ASSIGNED), None)
        active = next((k for k, a, b, _ in h if b == TaskState.ACTIVE), None)
        if assigned is None or active is None:
            continue
        phases["queue"].append(assigned - t.released)
        phases["gather"].append(active - assigned)
        phases["execute"].append(t.completed_at - active)
        by_type[t.type.name] += 1
    m = info["metrics"]
    return {"sc": sc, "cf": cf, "seed": seed, "tasks": m["tasks_completed"], "latency": m["mean_latency"],
            "kwh": m["energy_used_kwh"], "phases": {k: float(np.mean(v)) for k, v in phases.items()},
            "by_type": dict(by_type), "rt": dict(rt)}


def boot(d, n=10000):
    d = np.asarray(d, float)
    rng = np.random.default_rng(0)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


if __name__ == "__main__":
    sc, ca, cb, seeds = sys.argv[1], sys.argv[2], sys.argv[3], [int(s) for s in sys.argv[4].split(",")]
    procs = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    with Pool(procs) as p:
        res = p.map(episode, [(sc, c, s) for s in seeds for c in (ca, cb)])
    R = {(r["cf"], r["seed"]): r for r in res}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "throughput_%s_%s_%s.json" % (sc, ca, cb))
    json.dump(res, open(out, "w"))
    print("%s: %s vs %s, seeds %s (paired difference %s - %s, mean [95%% CI])" % (sc, cb, ca, seeds, cb, ca))

    def line(name, f):
        d = [f(R[(cb, s)]) - f(R[(ca, s)]) for s in seeds]
        m, lo, hi = boot(d)
        a = np.mean([f(R[(ca, s)]) for s in seeds])
        print("  %-28s %s %8.1f   diff %+8.1f [%+.1f, %+.1f]" % (name, ca, a, m, lo, hi))
    line("tasks completed", lambda r: r["tasks"])
    line("mean latency", lambda r: r["latency"])
    line("energy kWh", lambda r: r["kwh"])
    for k in ("queue", "gather", "execute"):
        line("per task: " + k, lambda r, k=k: r["phases"].get(k, 0))
    keys = sorted({k for r in res for k in r["rt"]})
    for k in keys:
        line("robot-ticks: " + k, lambda r, k=k: r["rt"].get(k, 0))
    types = sorted({k for r in res for k in r["by_type"]})
    for k in types:
        line("completed: " + k, lambda r, k=k: r["by_type"].get(k, 0))
