"""Reproduce the model 2.1 hold-and-wait gridlock, and show that model 2.2 removes it.

    python experiments/checks/known_gridlock.py            # both known episodes, models 2.1 and 2.2

Episodes (found in the first order-scheduling evaluation): base order scenario
(experiments/orders/factors.py), rule PLB, seed 1017 (600 ticks); and rate 0.05, PLB, seed 6
(3000 ticks). Under model 2.1, around tick 150 of the first episode a ``large`` task's second picking
robot, which works from a cell beside the shelf, waits on that cell for its crew, but the cell is
another task's work cell. That task's loaded AGV and picking robot wait next to it in the aisle and
block the route of the first task's other picking robot; no robot of these crews moves again and
the order layer reports suspected stalls. Model 2.2 keeps helpers off other tasks' work cells
(docs/semantics.md, "Model 2.2"). Exit status 0 iff 2.1 stalls and 2.2 does not, on both episodes.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "orders"))
import factors  # noqa: E402
from battery_tarware.orders.run import run_episode  # noqa: E402

EPISODES = [({}, "plb", 1017, 600), ({"rate": 0.05}, "plb", 6, 3000)]


def run(model, kw, rule, seed, ticks, verbose=True):
    cfg = factors.build(**kw)
    cfg["model"] = model
    res = run_episode(cfg, rule, seed, ticks, 0, record=False)
    stalls = [e for e in res["sim"].events.events if e.type == "stall_suspected"]
    print("model %s  %-12s %s seed %-5d  %d suspected stalls, %d orders completed"
          % (model, kw or "base", rule, seed, len(stalls), res["metrics"]["order"]["completed"]))
    if verbose:
        for e in stalls[:4]:
            d = e.data
            print("   tick %d  %s  %s  %s  last progress %d" % (e.tick, d["task_id"], d["wait_reason"],
                                                               d["blocking_resource"], d["last_progress_tick"]))
            for c in d["snapshot"]["crew"]:
                print("        %-16s %-9s at %s -> %s" % (c["name"], c["role"], tuple(c["cell"]), tuple(c["target"])))
    return len(stalls)


def main():
    ok = True
    for kw, rule, seed, ticks in EPISODES:
        ok &= run("2.1", kw, rule, seed, ticks) > 0
        ok &= run("2.2", kw, rule, seed, ticks) == 0
    print("2.1 GRIDLOCK REPRODUCED, 2.2 FREE OF IT" if ok else "UNEXPECTED OUTCOME")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
