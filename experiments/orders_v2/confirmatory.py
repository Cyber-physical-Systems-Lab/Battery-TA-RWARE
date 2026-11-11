"""CONFIRMATORY run of the second evaluation: exactly the protocol in protocol_v2.json, on its fresh
pre-registered seeds, each hypothesis judged by its decision rule.

    python experiments/orders_v2/confirmatory.py [--jobs 30]

Refuses to run if protocol_v2.json has uncommitted changes, and records the protocol's git blob.
Configurations: ``{"workload": {...}}`` (experiments/orders_v2/workloads.build arguments) or
``{"factors": {...}, "model": "2.1"}`` (the first study's experiments/orders/factors.build).
Hypotheses: ``kind: rules`` (rule vs reference rule within one configuration) or ``kind: models``
(the same rule in configuration ``config`` vs ``reference_config``, matched by seed; used for
model 2.2 vs 2.1). Directions: higher / lower (CI excludes 0 in that direction) or equivalent
(CI within +-margin). CIs: percentile bootstrap over seeds at the protocol's (Bonferroni) level.
The first study's protocol and results (experiments/orders/) are not touched.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "orders"))
import factors  # noqa: E402
import workloads  # noqa: E402
from battery_tarware.orders.evaluation import bootstrap_ci, run_grid, write_rows  # noqa: E402

PROTOCOL = os.path.join(HERE, "protocol_v2.json")
OUT = os.path.join(HERE, "results", "confirmatory")


def build(spec, ticks):
    if "workload" in spec:
        return workloads.build(max_steps=ticks, **spec["workload"])
    cfg = factors.build(max_steps=ticks, **spec["factors"])
    cfg["model"] = spec["model"]
    return cfg


def judge(h, d, level):
    lo, hi = bootstrap_ci(d, level=level)
    if h["direction"] == "higher":
        ok = lo > 0
    elif h["direction"] == "lower":
        ok = hi < 0
    else:
        ok = -h["margin"] < lo and hi < h["margin"]
    return lo, hi, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    if subprocess.run(["git", "status", "--porcelain", PROTOCOL], capture_output=True, text=True).stdout.strip():
        sys.exit("refusing to run: %s has uncommitted changes (commit the protocol first)" % PROTOCOL)
    blob = subprocess.run(["git", "rev-parse", "HEAD:experiments/orders_v2/protocol_v2.json"], capture_output=True,
                          text=True, cwd=os.path.join(HERE, "..", "..")).stdout.strip()
    proto = json.load(open(PROTOCOL))
    cfgs = {name: build(spec, proto["ticks"]) for name, spec in proto["configs"].items()}
    rows = []
    for name, rules in proto["rules"].items():               # only the rules each configuration needs
        rows += run_grid({name: cfgs[name]}, rules, proto["seeds"], proto["ticks"], proto["warmup"], jobs=a.jobs)
    write_rows(rows, os.path.join(OUT, "runs.csv"))
    ok_rows = [r for r in rows if not r.get("error")]
    by = {(r["config"], r["rule"], int(r["seed"])): r for r in ok_rows}
    results = []
    for h in proto["hypotheses"]:
        if h["kind"] == "rules":
            pa = [(h["config"], h["reference"]), (h["config"], h["rule"])]
        else:
            pa = [(h["reference_config"], h["rule"]), (h["config"], h["rule"])]
        pairs = [(by[pa[0] + (s,)].get(h["metric"]), by[pa[1] + (s,)].get(h["metric"])) for s in proto["seeds"]
                 if pa[0] + (s,) in by and pa[1] + (s,) in by]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        res = dict(h, n=len(pairs), supported=None)
        if len(pairs) > 1:
            ref = np.array([p[0] for p in pairs], float)
            val = np.array([p[1] for p in pairs], float)
            lo, hi, ok = judge(h, val - ref, proto["ci_level"])
            res.update(ref_mean=float(ref.mean()), mean=float(val.mean()), diff=float((val - ref).mean()),
                       ci_lo=lo, ci_hi=hi, supported=bool(ok), better=int(((val - ref) > 0).sum()))
        results.append(res)
    write_rows(results, os.path.join(OUT, "hypotheses.csv"))
    errors = [r for r in rows if r.get("error")]
    stalls = {c: sum(int(r.get("stalls") or 0) > 0 for r in ok_rows if r["config"] == c) for c in proto["configs"]}
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# Second evaluation, CONFIRMATORY run\n\nprotocol blob %s; seeds %s; %d ticks, warm-up %d; %d runs, "
                "%d errors; CI level %.5f (Bonferroni over %d hypotheses)\n\n"
                "| id | hypothesis | reference mean | mean | diff [CI] | seeds with diff > 0 | supported |\n"
                "|---|---|---|---|---|---|---|\n" % (blob, "%d-%d" % (proto["seeds"][0], proto["seeds"][-1]), proto["ticks"],
                                                     proto["warmup"], len(rows), len(errors), proto["ci_level"],
                                                     len(proto["hypotheses"])))
        for r in results:
            f.write("| %s | %s | %s | %s | %s | %s | %s |\n" % (
                r["id"], r["statement"], "%.4g" % r["ref_mean"] if "ref_mean" in r else "–",
                "%.4g" % r["mean"] if "mean" in r else "–",
                "%+.3g [%+.3g, %+.3g]" % (r["diff"], r["ci_lo"], r["ci_hi"]) if "diff" in r else "–",
                "%d/%d" % (r["better"], r["n"]) if "better" in r else "–",
                {True: "yes", False: "no", None: "not evaluable"}[r["supported"]]))
        f.write("\nRuns with a suspected stall, per configuration: %s\n" % stalls)
        for e in errors:
            f.write("- ERROR %s %s %s: %s\n" % (e["config"], e["rule"], e["seed"], e["error"]))
    print(open(os.path.join(OUT, "summary.md")).read())


if __name__ == "__main__":
    main()
