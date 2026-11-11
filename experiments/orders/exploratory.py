"""EXPLORATORY evaluation of the nine order-scheduling rules (+ WSP ablation).

    python experiments/orders/exploratory.py [--seeds 10] [--jobs 32] [--ticks 3000]

One-factor-at-a-time around the base scenario (factors.py): arrival rate, task mix, coalition
requirements (package weight), fleet heterogeneity, battery and docks, layout. Every rule runs on the
same seeds (paired: identical arrivals, inventory and robot placement). Outputs in
experiments/orders/results/exploratory/: runs.csv (one row per run), paired.csv (paired differences
to FIFO with bootstrap 95% CIs) and summary.md. Exploratory: no hypothesis was fixed in advance; use
the results to choose hypotheses for the confirmatory protocol, which runs on fresh seeds.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import factors  # noqa: E402
from battery_tarware.orders.evaluation import (PRIMARY, SECONDARY, markdown_table, paired, run_grid,  # noqa: E402
                                               write_rows)
from battery_tarware.orders.scheduler import RULE_NAMES  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "exploratory")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    cfgs = factors.configs(a.ticks)
    t0 = time.time()
    rows = run_grid(cfgs, RULE_NAMES, range(a.seeds), a.ticks, a.warmup, jobs=a.jobs,
                    out_csv=os.path.join(a.out, "runs.csv"))
    pr = paired(rows, "fifo", PRIMARY + SECONDARY)
    write_rows(pr, os.path.join(a.out, "paired.csv"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    errors = [r for r in rows if r.get("error")]
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("# Exploratory order-scheduling evaluation\n\ncommit %s; seeds 0-%d; %d ticks, warmup %d; %d runs, "
                "%d errors; %.0f s\n\nCells: rule mean (paired difference to FIFO [95%% bootstrap CI]); bold = CI "
                "excludes 0.\n\n" % (commit, a.seeds - 1, a.ticks, a.warmup, len(rows), len(errors), time.time() - t0))
        for c in cfgs:
            f.write("## %s\n\n" % c)
            f.write(markdown_table(rows, pr, c, PRIMARY) + "\n")
        for e in errors:
            f.write("- ERROR %s %s %s: %s\n" % (e["config"], e["rule"], e["seed"], e["error"]))
    print(os.path.join(a.out, "summary.md"), len(rows), "runs", len(errors), "errors", "%.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
