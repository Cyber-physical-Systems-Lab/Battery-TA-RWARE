"""Paired, seeded evaluation of order schedulers.

    rows = run_grid({"base": scenario_dict}, rules=RULE_NAMES, seeds=range(10), max_steps=3000, warmup=300, jobs=8)
    table = paired(rows, ref="fifo")              # per config, rule and metric: mean paired difference + 95% CI

Every (config, rule, seed) run uses the same scenario and seed for every rule, so arrivals, initial
inventory and robot placement are identical (the order generator's random draws do not depend on
the simulation state). Differences are paired by seed; the CI is a percentile bootstrap over seeds.
A run that raises is kept as a row with ``error`` set (a simulator/configuration error), never dropped.
"""
from __future__ import annotations

import csv
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

PRIMARY = ("throughput", "lead_time_mean", "on_time_rate", "energy_wh_per_order", "coalition_waiting_ratio",
           "unfinished")
SECONDARY = ("completion_rate", "lead_time_p95", "tardiness_mean", "energy_wh_per_delivery", "utilization",
             "jain", "distance_per_delivery", "congestion_share", "blocked_moves_per_1000", "formation_mean",
             "assignment_latency_mean", "runtime_ms_mean", "runtime_ms_p95", "candidates_mean", "charge_sessions",
             "dock_queue_ticks", "unfinished_age_mean", "rejected")
CAUSES = ("sequencing", "coalition_unavailable", "charging_contention", "breakdown", "battery", "congestion",
          "coalition_waiting", "travel", "execution", "blocked_dependency", "blocked_inventory")


def flatten(m: Dict) -> Dict:
    o, s, c, r, y, b = m["order"], m["scheduling"], m["coalition"], m["robot"], m["system"], m["bottlenecks"]
    row = {
        "throughput": o["throughput_per_1000"], "completion_rate": o["completion_rate"],
        "lead_time_mean": o["lead_time_mean"], "lead_time_p95": o["lead_time_p95"], "on_time_rate": o["on_time_rate"],
        "tardiness_mean": o["tardiness"]["mean"], "unfinished": o["unfinished"], "unfinished_age_mean": o["unfinished_age_mean"],
        "rejected": sum(1 for u in b["unsuccessful_orders"] if str(u.get("reason") or "").startswith("rejected")),
        "energy_wh_per_order": y["energy_wh_per_order"], "energy_wh_per_delivery": y["energy_wh_per_delivery"],
        "coalition_waiting_ratio": c["waiting_ratio"], "formation_mean": c["formation_time"]["mean"],
        "utilization": r["utilization_mean"], "jain": r["utilization_jain"], "distance_per_delivery": y["distance_per_delivery"],
        "congestion_share": y["congestion_waiting_share"], "blocked_moves_per_1000": y["congestion_blocked_moves_per_1000_robot_ticks"],
        "assignment_latency_mean": s["assignment_latency"]["mean"], "runtime_ms_mean": s["runtime_ms"]["mean"],
        "runtime_ms_p95": s["runtime_ms"]["p95"], "runtime_ms_p99": s["runtime_ms"]["p99"],
        "runtime_ms_p50": s["runtime_ms"]["p50"], "runtime_ms_max": s["runtime_ms"]["max"],
        "deadline_violation_rate": s.get("deadline_violation_rate"),
        "deadline_violations": s.get("deadline_violations"),
        "candidates_mean": s["candidates_per_tick"]["mean"],
        "charge_sessions": r["charge_sessions"], "dock_queue_ticks": r["dock_queue_ticks"], "stranded": r["stranded"],
        "retries": s["retries"], "stalls": s["stalls_suspected"], "wall_seconds": m["meta"].get("wall_seconds"),
    }
    share = b["task_share_by_cause"]
    for k in CAUSES:
        row["cause_" + k] = share.get(k, 0.0)
    return row


def run_one(job: Dict) -> Dict:
    from battery_tarware.orders.run import run_episode
    base = {"config": job["config"], "rule": job["rule"], "seed": job["seed"], "error": None}
    try:
        res = run_episode(job["scenario"], job["rule"], job["seed"], job["max_steps"], job["warmup"],
                          scheduler_kwargs=job.get("scheduler_kwargs"))
        row = {**base, **flatten(res["metrics"]), "robots": res["metrics"]["meta"]["robots"],
               "orders_arrived": res["metrics"]["order"]["orders_arrived"]}
        if job.get("trace_dir"):
            from battery_tarware.orders.trace import _json_default
            os.makedirs(job["trace_dir"], exist_ok=True)
            with open(os.path.join(job["trace_dir"], "%s_%s_%d.json" % (job["config"], job["rule"], job["seed"])), "w") as f:
                json.dump(res["trace"], f, default=_json_default, separators=(",", ":"))
        return row
    except Exception as e:                                     # a system / configuration error is a result
        return {**base, "error": "%s: %s" % (type(e).__name__, e), "traceback": traceback.format_exc(limit=3)}


def run_grid(configs: Dict[str, Dict], rules: Sequence[str], seeds: Iterable[int], max_steps: int, warmup: int = 0,
             jobs: int = 1, out_csv: Optional[str] = None, trace_dir: Optional[str] = None,
             trace_seeds: Sequence[int] = (), scheduler_kwargs: Optional[Dict] = None) -> List[Dict]:
    work = [{"config": c, "scenario": sc, "rule": r, "seed": s, "max_steps": max_steps, "warmup": warmup,
             "scheduler_kwargs": scheduler_kwargs, "trace_dir": trace_dir if s in trace_seeds else None}
            for c, sc in configs.items() for r in rules for s in seeds]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            rows = list(ex.map(run_one, work, chunksize=1))
    else:
        rows = [run_one(w) for w in work]
    if out_csv:
        write_rows(rows, out_csv)
    return rows


def write_rows(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r if k != "traceback"))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_rows(path: str) -> List[Dict]:
    def conv(v):
        if v in ("", "None"):
            return None
        try:
            return float(v)
        except ValueError:
            return v
    with open(path) as f:
        return [{k: (conv(v) if k not in ("config", "rule", "error") else (v or None)) for k, v in r.items()}
                for r in csv.DictReader(f)]


def bootstrap_ci(d: np.ndarray, n_boot: int = 10000, seed: int = 0, level: float = 0.95):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired(rows: List[Dict], ref: str = "fifo", metrics: Sequence[str] = PRIMARY, n_boot: int = 10000,
           level: float = 0.95) -> List[Dict]:
    ok = [r for r in rows if not r.get("error")]
    configs = list(dict.fromkeys(r["config"] for r in ok))
    rules = list(dict.fromkeys(r["rule"] for r in ok))
    by = {(r["config"], r["rule"], int(r["seed"])): r for r in ok}
    out = []
    for c in configs:
        seeds = sorted({int(r["seed"]) for r in ok if r["config"] == c})
        for rule in rules:
            for m in metrics:
                pairs = [(by[(c, ref, s)].get(m), by[(c, rule, s)].get(m)) for s in seeds
                         if (c, ref, s) in by and (c, rule, s) in by]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                if not pairs:
                    continue
                a = np.array([p[0] for p in pairs], float)
                b = np.array([p[1] for p in pairs], float)
                d = b - a
                lo, hi = bootstrap_ci(d, n_boot, level=level) if rule != ref and len(d) > 1 else (0.0, 0.0)
                out.append({"config": c, "rule": rule, "metric": m, "n": len(d), "ref_mean": float(a.mean()),
                            "rule_mean": float(b.mean()), "diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                            "rule_better": int(sum(_better(m, x) for x in d)), "excludes_zero": bool(lo > 0 or hi < 0)})
    return out


LOWER_IS_BETTER = {"lead_time_mean", "lead_time_p95", "energy_wh_per_order", "energy_wh_per_delivery",
                   "coalition_waiting_ratio", "unfinished", "tardiness_mean", "unfinished_age_mean",
                   "distance_per_delivery", "congestion_share", "blocked_moves_per_1000", "assignment_latency_mean",
                   "runtime_ms_mean", "runtime_ms_p95", "formation_mean", "dock_queue_ticks"}


def _better(metric: str, d: float) -> bool:
    return d < 0 if metric in LOWER_IS_BETTER else d > 0


def markdown_table(rows: List[Dict], paired_rows: List[Dict], config: str, metrics: Sequence[str] = PRIMARY,
                   ref: str = "fifo") -> str:
    """Per rule: mean of each metric and the paired difference to ``ref`` with its 95% CI."""
    rules = list(dict.fromkeys(r["rule"] for r in rows if r["config"] == config and not r.get("error")))
    idx = {(p["rule"], p["metric"]): p for p in paired_rows if p["config"] == config}
    head = "| rule | " + " | ".join(metrics) + " |\n|---|" + "---|" * len(metrics) + "\n"
    body = []
    for rule in rules:
        cells = []
        for m in metrics:
            p = idx.get((rule, m))
            if p is None:
                cells.append("–")
            elif rule == ref:
                cells.append("%s" % _f(p["rule_mean"]))
            else:
                mark = "**" if p["excludes_zero"] else ""
                cells.append("%s (%s%+.3g [%+.3g, %+.3g]%s)" % (_f(p["rule_mean"]), mark, p["diff"], p["ci_lo"],
                                                                 p["ci_hi"], mark))
        body.append("| %s | %s |" % (rule, " | ".join(cells)))
    return head + "\n".join(body) + "\n"


def _f(x) -> str:
    return "–" if x is None else ("%.3g" % x)
