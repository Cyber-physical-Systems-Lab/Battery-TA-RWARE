"""Scalability of the order schedulers: scheduler compute time and wall time per tick as the
warehouse, fleet and order rate grow together.

    python experiments/orders/scalability.py [--seeds 3] [--ticks 1500] [--jobs 30]

Sizes: S (basic 2x3 rack blocks, 9 robots, 0.09 orders/tick), M (3x4, 18 robots, 0.18),
L (4x6, 36 robots, 0.36). Fleet composition 2:1 AGV : picking robot as in basic. Writes
experiments/orders/results/scalability/{runs.csv,summary.md}.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import factors  # noqa: E402
from battery_tarware.orders.evaluation import run_grid  # noqa: E402
from battery_tarware.orders.scheduler import RULE_NAMES  # noqa: E402

SIZES = {"S": (2, 3, 2, 4, 6, 3, 0.09), "M": (3, 4, 3, 6, 12, 6, 0.18), "L": (4, 6, 4, 8, 24, 12, 0.36)}
OUT = os.path.join(os.path.dirname(__file__), "results", "scalability")


def scenario(size, ticks):
    rr, rc, goals, docks, agv, pick, rate = SIZES[size]
    cfg = factors.build(rate=rate, max_steps=ticks)
    cfg["name"] = "orders_scale_%s" % size
    cfg["map"] = {"generator": {"rack_rows": rr, "rack_cols": rc, "rack_height": 4, "goals": goals, "docks": docks}}
    cfg["fleet"] = [{"type": "agv", "count": agv, "initial_soc": [0.5, 1.0]},
                    {"type": "picking_robot", "count": pick, "initial_soc": [0.5, 1.0]}]
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    rows = run_grid({s: scenario(s, a.ticks) for s in SIZES}, RULE_NAMES, range(a.seeds), a.ticks, 150, jobs=a.jobs,
                    out_csv=os.path.join(OUT, "runs.csv"))
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# Scalability (seeds 0-%d, %d ticks)\n\nScheduler time per tick (ms, mean / p95), simulation wall "
                "time per tick (ms, whole step incl. recording), candidate pairs per tick, throughput per 1000 ticks.\n\n"
                "| size | robots | rule | sched ms mean | sched ms p95 | wall ms/tick | pairs/tick | throughput |\n"
                "|---|---|---|---|---|---|---|---|\n" % (a.seeds - 1, a.ticks))
        for s in SIZES:
            for rule in RULE_NAMES:
                rs = [r for r in rows if r["config"] == s and r["rule"] == rule and not r.get("error")]
                if not rs:
                    f.write("| %s | – | %s | error | | | | |\n" % (s, rule))
                    continue
                mean = lambda k: float(np.mean([r[k] for r in rs]))
                f.write("| %s | %d | %s | %.2f | %.2f | %.1f | %.1f | %.1f |\n" % (
                    s, rs[0]["robots"], rule, mean("runtime_ms_mean"), mean("runtime_ms_p95"),
                    1000 * mean("wall_seconds") / a.ticks, mean("candidates_mean"), mean("throughput")))
        for r in rows:
            if r.get("error"):
                f.write("- ERROR %s %s %s: %s\n" % (r["config"], r["rule"], r["seed"], r["error"]))
    print(open(os.path.join(OUT, "summary.md")).read())


if __name__ == "__main__":
    main()
