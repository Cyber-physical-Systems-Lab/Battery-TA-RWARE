"""Reference numbers for the built-in tiers (README table).

    python benchmarks/baselines.py            # full-length episodes, seeds 0-4
"""
import sys
import time

import numpy as np

import battery_tarware as bt

SEEDS = range(5)
POLICIES = {"greedy-fifo": lambda: bt.GreedyCoalitionCoordinator("fifo"),
            "greedy-priority": lambda: bt.GreedyCoalitionCoordinator("priority"),
            "random (masked)": lambda: bt.RandomCoordinator(seed=0, noop_prob=0.5)}

print("| scenario | policy | tasks completed | mean latency (ticks) | late | energy (kWh) | stranded | sim speed |")
print("|---|---|---|---|---|---|---|---|")
for name in sys.argv[1:] or sorted(bt.SCENARIOS, key=["basic", "intermediate", "advanced"].index):
    for pname, make in POLICIES.items():
        rows, t0, ticks = [], time.time(), 0
        for seed in SEEDS:
            m = bt.run_episode(bt.make(name), make(), seed=seed)
            rows.append(m)
            ticks += m["tick"]
        f = lambda k: np.array([r[k] if r[k] is not None else np.nan for r in rows], float)
        print("| %s | %s | %.1f ± %.1f | %.0f | %.1f | %.2f | %.1f | %.0f ticks/s |" % (
            name, pname, f("tasks_completed").mean(), f("tasks_completed").std(), np.nanmean(f("mean_latency")),
            f("late_tasks").mean(), f("energy_used_kwh").mean(), f("stranded").mean(), ticks / (time.time() - t0)),
            flush=True)
