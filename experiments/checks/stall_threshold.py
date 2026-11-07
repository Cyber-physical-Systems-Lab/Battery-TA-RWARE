"""Validate the stall threshold on successful task-only baseline traces (greedy FIFO coordinator).

    python experiments/checks/stall_threshold.py [ticks] [seeds]

For every built-in scenario and seed, the progress monitor runs read-only next to the greedy
coordinator. Reported per scenario: the longest eligible no-progress streak of any task (the stall
clock), the number of tasks whose streak reached 50 / 100 ticks, the stalls a 100-tick threshold
would report, and the ticks spent in each legitimate waiting class. A threshold is safe for
automatic retries only if successful runs never reach it.
"""
import sys

import battery_tarware as bt
from battery_tarware.orders.monitor import ProgressMonitor

ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
print("scenario      seeds  ticks  tasks_done  max_streak  streak>=50  streak>=100  reports@100  waits")
for name in ("basic", "intermediate", "advanced"):
    best, n50, n100, reps, done, waits = 0, 0, 0, [], 0, {}
    for seed in seeds:
        env = bt.make(name, max_steps=ticks)
        sim = env.unwrapped.sim
        env.reset(seed=seed)
        mon = ProgressMonitor(sim, 100)
        co = bt.GreedyCoalitionCoordinator("fifo")
        peak = {}
        for _ in range(ticks):
            _, _, term, trunc, _ = env.step(co.act(env))
            reps += [(seed, r.sim_task, r.wait_reason, r.blocking_resource) for r in mon.update()]
            for tid, st in mon.state.items():
                peak[tid] = max(peak.get(tid, 0), st.clock)
            if term or trunc:
                break
        best = max([best] + list(peak.values()))
        n50 += sum(v >= 50 for v in peak.values())
        n100 += sum(v >= 100 for v in peak.values())
        done += sim.metrics()["tasks_completed"]
        for st in mon.state.values():
            for k, v in st.waits.items():
                waits[k] = waits.get(k, 0) + v
    tot = sum(waits.values())
    w = "  ".join("%s %.1f%%" % (k, 100 * v / tot) for k, v in sorted(waits.items()))
    print("%-13s %5d %6d %11d %11d %11d %12d %12d  %s" % (name, len(seeds), ticks, done, best, n50, n100, len(reps), w))
    for r in reps:
        print("   report: seed %d task %d %s (%s)" % r)
