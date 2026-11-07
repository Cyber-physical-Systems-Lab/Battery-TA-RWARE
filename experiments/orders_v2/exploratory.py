"""EXPLORATORY pass of the second order-scheduling evaluation (model 2.2, expanded warehouses).

    python experiments/orders_v2/exploratory.py [--seeds 2000-2009] [--jobs 30] [--ticks 3000]

Fresh seeds 2000-2009 (never used before; the first study used 0-9 and 1000-1019). No hypothesis is
fixed in advance: the results are used to choose the hypotheses of the second confirmatory protocol,
which runs on other fresh seeds. Two grids, every rule on every seed (paired):

G  generalization, model 2.2: basic / intermediate / advanced with the standard order profile;
   intermediate and advanced with the dag profile (larger dependency graphs) and with tight
   batteries; intermediate without role alternatives; basic with the first study's simple profile.
M  model 2.1 vs 2.2 on the first study's exploratory configurations (experiments/orders/factors.py:
   base and every alternative level), matched workloads: same configuration, rule and seed.

Outputs (experiments/orders_v2/results/exploratory/): runs.csv, summary.md.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "orders"))
import factors  # noqa: E402
import workloads  # noqa: E402
from battery_tarware.orders.evaluation import PRIMARY, bootstrap_ci, markdown_table, paired, run_grid  # noqa: E402
from battery_tarware.orders.scheduler import RULE_NAMES  # noqa: E402

OUT = os.path.join(HERE, "results", "exploratory")

G_CONFIGS = {
    "basic/standard": dict(base="basic", profile="standard"),
    "basic/simple": dict(base="basic", profile="simple"),
    "intermediate/standard": dict(base="intermediate", profile="standard"),
    "intermediate/dag": dict(base="intermediate", profile="dag"),
    "intermediate/tight": dict(base="intermediate", profile="standard", battery="tight"),
    "intermediate/no_alternatives": dict(base="intermediate", profile="standard", substitution=False),
    "advanced/standard": dict(base="advanced", profile="standard"),
    "advanced/dag": dict(base="advanced", profile="dag"),
    "advanced/tight": dict(base="advanced", profile="standard", battery="tight"),
}


def g_configs(ticks, model="2.2"):
    return {k: workloads.build(model=model, max_steps=ticks, **v) for k, v in G_CONFIGS.items()}


def m_configs(ticks):
    out = {}
    for name, cfg in factors.configs(ticks).items():
        for model in ("2.1", "2.2"):
            c = dict(cfg)
            c["model"] = model
            out["%s@%s" % (name, model)] = c
    return out


def model_pairs(rows, metrics=("throughput", "lead_time_mean", "energy_wh_per_order", "coalition_waiting_ratio",
                                "stalls", "runtime_ms_mean"), level=0.95):
    """Paired 2.2 - 2.1 differences per (configuration, rule, metric), matched by seed."""
    ok = [r for r in rows if not r.get("error")]
    by = {(r["config"], r["rule"], int(r["seed"])): r for r in ok}
    out = []
    names = sorted({c.rsplit("@", 1)[0] for c, _, _ in by})
    for n in names:
        for rule in RULE_NAMES:
            seeds = sorted({s for c, ru, s in by if c == n + "@2.1" and ru == rule})
            for m in metrics:
                pr = [(by[(n + "@2.1", rule, s)].get(m), by[(n + "@2.2", rule, s)].get(m)) for s in seeds
                      if (n + "@2.2", rule, s) in by]
                pr = [(a, b) for a, b in pr if a is not None and b is not None]
                if not pr:
                    continue
                a = np.array([p[0] for p in pr], float)
                b = np.array([p[1] for p in pr], float)
                d = b - a
                lo, hi = bootstrap_ci(d, level=level) if len(d) > 1 else (0.0, 0.0)
                out.append({"config": n, "rule": rule, "metric": m, "n": len(d), "m21": float(a.mean()),
                            "m22": float(b.mean()), "diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                            "identical": int((d == 0).sum())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2000-2009")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--grids", default="G,M")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = range(lo, hi + 1)
    cfgs = {}
    if "G" in a.grids:
        cfgs.update(g_configs(a.ticks))
    if "M" in a.grids:
        cfgs.update(m_configs(a.ticks))
    t0 = time.time()
    rows = run_grid(cfgs, RULE_NAMES, seeds, a.ticks, a.warmup, jobs=a.jobs, out_csv=os.path.join(OUT, "runs.csv"))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    errors = [r for r in rows if r.get("error")]
    write_summary(rows, commit, a, time.time() - t0, errors)
    print(os.path.join(OUT, "summary.md"), len(rows), "runs", len(errors), "errors")


def write_summary(rows, commit, a, secs, errors):
    L = ["# Second evaluation, EXPLORATORY pass", "",
         "commit %s; seeds %s; %d ticks, warm-up %d; %d runs, %d errors; %.0f s. Exploratory: no hypothesis "
         "was fixed in advance." % (commit, a.seeds, a.ticks, a.warmup, len(rows), len(errors), secs), "",
         "## G. Generalization (model 2.2): rule mean (paired difference to FIFO [95% CI]); bold = CI excludes 0", ""]
    g = [r for r in rows if "@" not in r["config"]]
    if g:
        pr = paired(g, "fifo", PRIMARY)
        for c in dict.fromkeys(r["config"] for r in g):
            L += ["### %s" % c, "", markdown_table(g, pr, c, PRIMARY)]
            st = sum(int(r.get("stalls") or 0) for r in g if r["config"] == c and not r.get("error"))
            L += ["suspected stalls (all rules and seeds): %d" % st, ""]
    m = [r for r in rows if "@" in r["config"]]
    if m:
        mp = model_pairs(m)
        L += ["## M. Model 2.2 - 2.1, matched (configuration, rule, seed): mean difference [95% CI] "
              "(identical seeds / n)", ""]
        for metric in ("throughput", "lead_time_mean", "stalls"):
            L += ["### %s" % metric, "", "| config | " + " | ".join(RULE_NAMES) + " |", "|---|" + "---|" * len(RULE_NAMES)]
            for c in dict.fromkeys(p["config"] for p in mp):
                cells = []
                for rule in RULE_NAMES:
                    p = [x for x in mp if x["config"] == c and x["rule"] == rule and x["metric"] == metric]
                    cells.append("%+.2f [%+.2f, %+.2f] (%d/%d)" % (p[0]["diff"], p[0]["ci_lo"], p[0]["ci_hi"],
                                                                  p[0]["identical"], p[0]["n"]) if p else "–")
                L.append("| %s | %s |" % (c, " | ".join(cells)))
            L.append("")
        for model in ("2.1", "2.2"):
            st = [r for r in m if r["config"].endswith("@" + model) and not r.get("error") and int(r.get("stalls") or 0) > 0]
            L.append("- model %s: %d of %d runs with a suspected stall%s" % (
                model, len(st), sum(1 for r in m if r["config"].endswith("@" + model)),
                (": " + ", ".join("%s %s %d" % (r["config"], r["rule"], int(r["seed"])) for r in st[:12])) if st else ""))
    L += [""] + ["- ERROR %s %s %s: %s" % (e["config"], e["rule"], e["seed"], e["error"]) for e in errors]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
