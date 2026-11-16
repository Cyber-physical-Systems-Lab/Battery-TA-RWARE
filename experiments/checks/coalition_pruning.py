"""Candidate pruning in ``feasible_coalitions`` against exhaustive generation.

    python experiments/checks/coalition_pruning.py

Runs the greedy coordinator on every built-in scenario (and intermediate with role alternatives) and,
at every tick, asks three generators for the crews of every open task: the default (6 candidates per
role, payload-aware for carriers, 2.1.1), the 2.1.0 rule (``legacy_pruning=True``) and the
exhaustive one (every eligible robot). Reported per scenario:

* queries, and how many have a feasible crew at all (exhaustive);
* false negatives: exhaustive finds a crew, the pruned generator finds none;
* different best crew: both find one, with a different estimated completion time;
* mean time per query.

Exit status 1 if the default generator has a false negative or a different best completion time.
"""
import sys
import time

import numpy as np

import battery_tarware as bt
from battery_tarware.coalition import feasible_coalitions

TICKS = 1500
SEEDS = (0, 1, 2)


def scenarios():
    for name in ("basic", "intermediate", "advanced"):
        yield name, bt.get_scenario(name)
    yield "intermediate+alternatives", bt.get_scenario("intermediate", substitution=True)
    for name in ("basic", "intermediate"):
        yield name + " (model 2.3)", bt.get_scenario(name, model="2.3")


def main() -> int:
    bad = 0
    print("%-28s %8s %9s | %-21s | %-21s | ms/query: default legacy exhaustive" % (
        "scenario", "queries", "feasible", "default: miss  differ", "legacy:  miss  differ"))
    for name, d in scenarios():
        n = feas = 0
        miss = {"default": 0, "legacy": 0}
        differ = {"default": 0, "legacy": 0}
        spent = {"default": 0.0, "legacy": 0.0, "all": 0.0}
        for seed in SEEDS:
            sim = bt.Simulator(bt.ScenarioConfig.from_dict(dict(d, max_steps=TICKS)))
            sim.reset(seed)
            co = bt.GreedyCoalitionCoordinator("fifo")
            idle = np.zeros(len(sim.robots), dtype=np.int64)
            for _ in range(TICKS):
                for t in sim.open_tasks():
                    if not t.joinable_roles():
                        continue
                    res = {}
                    for key, kw in (("default", {}), ("legacy", {"legacy_pruning": True}),
                                    ("all", {"per_role_candidates": len(sim.robots)})):
                        t0 = time.perf_counter()
                        res[key] = feasible_coalitions(sim, t, limit=1, energy_reserve=0.15, **kw)
                        spent[key] += time.perf_counter() - t0
                    n += 1
                    if not res["all"]:
                        assert not res["default"] and not res["legacy"]      # pruning never adds a crew
                        continue
                    feas += 1
                    for key in ("default", "legacy"):
                        if not res[key]:
                            miss[key] += 1
                        elif abs(res[key][0].est_ticks - res["all"][0].est_ticks) > 1e-9:
                            differ[key] += 1
                co.act(sim)
                sim.step(idle)
        bad += miss["default"] + differ["default"]
        print("%-28s %8d %9d | %13d %7d | %13d %7d | %18.3f %6.3f %10.3f" % (
            name, n, feas, miss["default"], differ["default"], miss["legacy"], differ["legacy"],
            1000 * spent["default"] / max(1, n), 1000 * spent["legacy"] / max(1, n), 1000 * spent["all"] / max(1, n)))
    print("default generator: %s" % ("OK" if not bad else "%d disagreements with exhaustive" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
