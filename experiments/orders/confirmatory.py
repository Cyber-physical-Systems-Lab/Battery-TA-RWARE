"""CONFIRMATORY evaluation: runs exactly the protocol in confirmatory_protocol.json on its fresh,
pre-registered seeds and evaluates each pre-registered hypothesis by its decision rule.

    python experiments/orders/confirmatory.py [--jobs 30]

The protocol (configs, rules, seeds, horizon, hypotheses, decision rules) must be committed before
this script is run; the script refuses to run if the protocol file has uncommitted changes, and
records the protocol's git blob hash with the results. Outputs: experiments/orders/results/confirmatory/.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import factors  # noqa: E402
from battery_tarware.orders.evaluation import paired, run_grid, write_rows  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL = os.path.join(HERE, "confirmatory_protocol.json")
OUT = os.path.join(HERE, "results", "confirmatory")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    dirty = subprocess.run(["git", "status", "--porcelain", PROTOCOL], capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit("refusing to run: %s has uncommitted changes (commit the protocol first)" % PROTOCOL)
    blob = subprocess.run(["git", "rev-parse", "HEAD:experiments/orders/confirmatory_protocol.json"],
                          capture_output=True, text=True, cwd=os.path.join(HERE, "..", "..")).stdout.strip()
    proto = json.load(open(PROTOCOL))
    cfgs = {name: factors.build(**levels, max_steps=proto["ticks"]) for name, levels in proto["configs"].items()}
    rows = run_grid(cfgs, proto["rules"], proto["seeds"], proto["ticks"], proto["warmup"], jobs=a.jobs,
                    out_csv=os.path.join(OUT, "runs.csv"))
    metrics = sorted({h["metric"] for h in proto["hypotheses"]})
    results = []
    for h in proto["hypotheses"]:
        pr = [p for p in paired(rows, h["reference"], (h["metric"],), level=proto["ci_level"])
              if p["config"] == h["config"] and p["rule"] == h["rule"]]
        p = pr[0] if pr else None
        ok = None
        if p is not None:
            if h["direction"] == "higher":
                ok = p["ci_lo"] > 0
            elif h["direction"] == "lower":
                ok = p["ci_hi"] < 0
            else:                                        # equivalence within +-margin
                ok = -h["margin"] < p["ci_lo"] and p["ci_hi"] < h["margin"]
        results.append({**h, **({k: p[k] for k in ("n", "ref_mean", "rule_mean", "diff", "ci_lo", "ci_hi")} if p else {}),
                        "supported": ok})
    write_rows(results, os.path.join(OUT, "hypotheses.csv"))
    errors = [r for r in rows if r.get("error")]
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("# Confirmatory order-scheduling evaluation\n\nprotocol blob %s; seeds %s; %d runs, %d errors; "
                "CI level %.4f (Bonferroni over %d hypotheses)\n\n"
                "| id | hypothesis | rule mean | reference mean | diff [CI] | supported |\n|---|---|---|---|---|---|\n"
                % (blob, proto["seeds"], len(rows), len(errors), proto["ci_level"], len(proto["hypotheses"])))
        for r in results:
            f.write("| %s | %s | %.4g | %.4g | %s | %s |\n" % (r["id"], r["statement"], r.get("rule_mean", float("nan")),
                    r.get("ref_mean", float("nan")),
                    "%+.3g [%+.3g, %+.3g]" % (r["diff"], r["ci_lo"], r["ci_hi"]) if "diff" in r else "–",
                    {True: "yes", False: "no", None: "not evaluable"}[r["supported"]]))
        for e in errors:
            f.write("- ERROR %s %s %s: %s\n" % (e["config"], e["rule"], e["seed"], e["error"]))
    print(open(os.path.join(OUT, "summary.md")).read())


if __name__ == "__main__":
    main()
