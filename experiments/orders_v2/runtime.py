"""Scheduler runtime and deadline compliance, per rule, as the problem grows (model 2.2).

    python experiments/orders_v2/runtime.py [--seeds 3] [--ticks 1500] [--jobs 8] [--deadline 20]

Families
  scale      basic warehouse family, fleet and order rate grown together: S 9 robots (2x3 rack
             blocks), M 18 (3x4), L 36 (4x6), XL 72 (6x8); standard order profile
  warehouse  intermediate and advanced warehouses x order profiles simple / standard / dag
             (dependency complexity)

Per (family, case, rule) over seeds: scheduler wall-clock time per tick (charging rule + assignment)
mean, median, P95, P99, max; share of ticks over the deadline (all ticks and decision ticks with at
least one eligible task); mean eligible tasks, free robots and candidate (task, coalition) pairs per
tick; throughput. Also P99 and violation rate binned by eligible tasks and by candidate pairs.
Wall-clock times depend on the machine and on its load: the script records the CPU model and runs
``--jobs`` episodes in parallel (default 8 of the machine's cores). The deadline is a reporting
target; the scheduler's algorithmic budget is count-based and never cut short by it.
Writes experiments/orders_v2/results/runtime/{runs.json,summary.md} (git-ignored).
"""
import argparse
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results", "runtime")
SCALE = {"S": (2, 3, 2, 4, 6, 3, 0.09), "M": (3, 4, 3, 6, 12, 6, 0.18), "L": (4, 6, 4, 8, 24, 12, 0.36),
         "XL": (6, 8, 6, 16, 48, 24, 0.72)}
ELIG_BINS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 10 ** 9)]
PAIR_BINS = [(0, 0), (1, 10), (11, 50), (51, 200), (201, 500), (501, 10 ** 9)]


def scenario(family, case, ticks):
    import workloads
    if family == "scale":
        rr, rc, goals, docks, agv, pick, rate = SCALE[case]
        cfg = workloads.build("basic", "2.2", "standard", rate=rate, max_steps=ticks)
        cfg["map"] = {"generator": {"rack_rows": rr, "rack_cols": rc, "rack_height": 4, "goals": goals, "docks": docks}}
        cfg["fleet"] = [{"type": "agv", "count": agv, "initial_soc": [0.5, 1.0]},
                        {"type": "picking_robot", "count": pick, "initial_soc": [0.5, 1.0]}]
        return cfg
    base, profile = case.split("/")
    return workloads.build(base, "2.2", profile, max_steps=ticks)


def one(job):
    from battery_tarware.orders.run import run_episode
    fam, case, rule, seed, ticks, deadline = job
    try:
        res = run_episode(scenario(fam, case, ticks), rule, seed, ticks, ticks // 10,
                          scheduler_kwargs={"deadline_ms": deadline}, record=True)
    except Exception as e:  # noqa: BLE001
        return {"job": job, "error": "%s: %s" % (type(e).__name__, e)}
    rt = np.array(res["trace"]["runtime"], dtype=float)          # tick, ms, pairs, eligible, free, charging ms
    m = res["metrics"]

    def binned(col, bins):
        out = []
        for lo, hi in bins:
            sel = rt[(rt[:, col] >= lo) & (rt[:, col] <= hi), 1]
            out.append([lo, hi, int(len(sel)), float(np.percentile(sel, 99)) if len(sel) else None,
                        float((sel > deadline).mean()) if len(sel) else None])
        return out
    dec = rt[rt[:, 3] > 0, 1]
    return {"job": job, "robots": m["meta"]["robots"], "ticks": len(rt),
            "ms": rt[:, 1].tolist() if fam == "scale" and case == "XL" else None,
            "mean": float(rt[:, 1].mean()), "p50": float(np.percentile(rt[:, 1], 50)),
            "p95": float(np.percentile(rt[:, 1], 95)), "p99": float(np.percentile(rt[:, 1], 99)),
            "max": float(rt[:, 1].max()), "viol": float((rt[:, 1] > deadline).mean()),
            "viol_dec": float((dec > deadline).mean()) if len(dec) else None, "n_viol": int((rt[:, 1] > deadline).sum()),
            "charging_ms_mean": float(rt[:, 5].mean()), "eligible": float(rt[:, 3].mean()), "free": float(rt[:, 4].mean()),
            "pairs": float(rt[:, 2].mean()), "pairs_max": int(rt[:, 2].max()), "eligible_max": int(rt[:, 3].max()),
            "throughput": m["order"]["throughput_per_1000"], "wall_s": m["meta"]["wall_seconds"],
            "by_eligible": binned(3, ELIG_BINS), "by_pairs": binned(2, PAIR_BINS)}


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main(argv=None):
    from battery_tarware.orders.scheduler import RULE_NAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--deadline", type=float, default=20.0)
    ap.add_argument("--families", default="scale,warehouse")
    a = ap.parse_args(argv)
    cases = []
    if "scale" in a.families:
        cases += [("scale", c) for c in SCALE]
    if "warehouse" in a.families:
        cases += [("warehouse", "%s/%s" % (b, p)) for b in ("intermediate", "advanced") for p in ("simple", "standard", "dag")]
    jobs = [(f, c, r, s, a.ticks, a.deadline) for f, c in cases for r in RULE_NAMES for s in range(a.seeds)]
    jobs.sort(key=lambda j: j[1] != "XL")                    # longest first
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(one, jobs, chunksize=1))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "runs.json"), "w") as f:
        json.dump({"cpu": cpu_name(), "cores": os.cpu_count(), "parallel_jobs": a.jobs, "deadline_ms": a.deadline,
                   "ticks": a.ticks, "seeds": a.seeds, "rows": rows}, f)
    write_summary(rows, a)
    print(os.path.join(OUT, "summary.md"))


def write_summary(rows, a):
    from battery_tarware.orders.scheduler import RULE_NAMES
    ok = [r for r in rows if not r.get("error")]
    L = ["# Scheduler runtime and deadline compliance (model 2.2)", "",
         "CPU: %s, %d cores; %d episodes in parallel; %d ticks per episode (first 10%% warm-up for "
         "throughput only; runtime over all ticks); seeds 0-%d; deadline %.0f ms (reporting target)."
         % (cpu_name(), os.cpu_count(), a.jobs, a.ticks, a.seeds - 1, a.deadline), "",
         "ms = scheduler wall-clock time per tick (charging rule + assignment). Violations: share of "
         "ticks over the deadline (all ticks / decision ticks, i.e. ticks with an eligible task); n = "
         "violating ticks summed over seeds. Pairs = feasible (task, coalition) candidates per tick.", "",
         "| family | case | robots | rule | mean | median | P95 | P99 | max | viol. % | viol. % (decision) | n viol. | eligible | pairs (max) | throughput |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    cases = list(dict.fromkeys((r["job"][0], r["job"][1]) for r in ok))
    for f, c in cases:
        for rule in RULE_NAMES:
            rs = [r for r in ok if r["job"][0] == f and r["job"][1] == c and r["job"][2] == rule]
            if not rs:
                continue
            mean = lambda k: float(np.mean([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else float("nan")
            L.append("| %s | %s | %d | %s | %.2f | %.2f | %.2f | %.2f | %.1f | %.2f | %.2f | %d | %.1f | %.0f (%d) | %.1f |" % (
                f, c, rs[0]["robots"], rule, mean("mean"), mean("p50"), mean("p95"), mean("p99"),
                max(r["max"] for r in rs), 100 * mean("viol"), 100 * mean("viol_dec"), sum(r["n_viol"] for r in rs),
                mean("eligible"), mean("pairs"), max(r["pairs_max"] for r in rs), mean("throughput")))
    L += ["", "## P99 (ms) and violation rate by eligible tasks per tick (all rules and seeds pooled per case)", "",
          "| case | " + " | ".join("%d-%s" % (lo, hi if hi < 10 ** 8 else "") for lo, hi in ELIG_BINS) + " |",
          "|---|" + "---|" * len(ELIG_BINS)]
    for f, c in cases:
        rs = [r for r in ok if r["job"][0] == f and r["job"][1] == c]
        cells = []
        for k in range(len(ELIG_BINS)):
            n = sum(r["by_eligible"][k][2] for r in rs)
            p = [r["by_eligible"][k][3] for r in rs if r["by_eligible"][k][3] is not None]
            v = [r["by_eligible"][k][4] * r["by_eligible"][k][2] for r in rs if r["by_eligible"][k][4] is not None]
            cells.append("–" if not n else "%.1f / %.1f%% (n=%d)" % (max(p), 100 * sum(v) / n, n))
        L.append("| %s %s | %s |" % (f, c, " | ".join(cells)))
    L += ["", "## P99 (ms) and violation rate by candidate pairs per tick", "",
          "| case | " + " | ".join("%d-%s" % (lo, hi if hi < 10 ** 8 else "") for lo, hi in PAIR_BINS) + " |",
          "|---|" + "---|" * len(PAIR_BINS)]
    for f, c in cases:
        rs = [r for r in ok if r["job"][0] == f and r["job"][1] == c]
        cells = []
        for k in range(len(PAIR_BINS)):
            n = sum(r["by_pairs"][k][2] for r in rs)
            p = [r["by_pairs"][k][3] for r in rs if r["by_pairs"][k][3] is not None]
            v = [r["by_pairs"][k][4] * r["by_pairs"][k][2] for r in rs if r["by_pairs"][k][4] is not None]
            cells.append("–" if not n else "%.1f / %.1f%% (n=%d)" % (max(p), 100 * sum(v) / n, n))
        L.append("| %s %s | %s |" % (f, c, " | ".join(cells)))
    errs = [r for r in rows if r.get("error")]
    L += [""] + ["- ERROR %s: %s" % (r["job"], r["error"]) for r in errs]
    with open(os.path.join(OUT, "summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
