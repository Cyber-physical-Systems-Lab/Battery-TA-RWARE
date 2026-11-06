"""Order-, scheduling-, robot-, coalition- and system-level metrics and failure diagnostics.

``evaluate(trace)`` takes the dict of ``TraceRecorder.to_dict()`` (event log + per-tick frames), so
every algorithm is measured by the same code from the same record. Definitions are in
``docs/metrics.md``; the conventions:

* time in ticks; horizon ``T`` = last recorded tick; rates use the window ``(warmup, T]``;
* work unfinished at ``T`` is censored: excluded from lead time and tardiness, reported by count
  and age, never counted as failed;
* "robot-ticks" count one robot for one tick; shares are over all robot-ticks of the run.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from battery_tarware.orders.reports import order_metrics
from battery_tarware.orders.trace import ACTIVITIES, CAUSES


class _E:                                            # event view over the trace's event rows
    __slots__ = ("tick", "type", "robots", "task", "data")

    def __init__(self, row):
        self.tick, self.type, self.robots, self.task, self.data = row


def _stats(x) -> Dict:
    x = [v for v in x if v is not None]
    if not x:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    a = np.asarray(x, dtype=float)
    return {"n": len(x), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def _jain(x) -> Optional[float]:
    a = np.asarray(x, dtype=float)
    return float(a.sum() ** 2 / (len(a) * (a ** 2).sum())) if len(a) and (a ** 2).sum() > 0 else None


def evaluate(trace: Dict, warmup: int = 0) -> Dict:
    ev = [_E(r) for r in trace["events"]]
    meta = trace["meta"]
    T = meta["first_tick"] + meta["ticks"] - 1
    frames = trace["frames"]
    n_rob = len(meta["robots"])
    out: Dict = {"meta": {"scenario": meta["scenario"], "seed": meta["seed"], "rule": meta.get("rule"),
                          "horizon": T, "warmup": warmup, "robots": n_rob}}

    # ───────────────────────────── order level
    o = order_metrics(ev, T, warmup)
    arrival, due, kind, prio, closed = {}, {}, {}, {}, {}
    for e in ev:
        if e.type == "order_arrived":
            d = e.data
            arrival[d["order_id"]], due[d["order_id"]] = e.tick, d.get("due")
            kind[d["order_id"]], prio[d["order_id"]] = d.get("kind"), d.get("priority")
        elif e.type in ("order_completed", "order_failed", "order_cancelled"):
            closed[e.data["order_id"]] = (e.type[6:], e.tick)
    comp = [k for k in arrival if closed.get(k, ("",))[0] == "completed"]
    tard = [max(0, closed[k][1] - due[k]) for k in comp if due[k] is not None]
    by_kind: Dict[str, Dict] = {}
    for k in arrival:
        b = by_kind.setdefault(kind[k] or "?", {"arrived": 0, "completed": 0, "lead": []})
        b["arrived"] += 1
        if k in comp:
            b["completed"] += 1
            b["lead"].append(closed[k][1] - arrival[k])
    by_prio: Dict[str, List] = {}
    for k in comp:
        by_prio.setdefault(str(prio[k]), []).append(closed[k][1] - arrival[k])
    open_s = [v for v in trace["series"].get("open_orders", []) if v is not None]
    slope = None
    if len(open_s) > 10:
        x = np.arange(len(open_s))
        slope = float(np.polyfit(x, np.asarray(open_s, float), 1)[0] * 1000)
    o.update({
        "completion_rate": (o["completed"] / o["orders_arrived"]) if o["orders_arrived"] else None,
        "tardiness": _stats(tard),
        "late_completed": sum(1 for v in tard if v > 0),
        "deadline_miss_rate": (1 - o["on_time_rate"]) if o["on_time_rate"] is not None else None,
        "wip_mean": float(np.mean(open_s)) if open_s else None,
        "backlog_slope_per_1000": slope,
        "by_kind": {k: {"arrived": v["arrived"], "completed": v["completed"],
                        "lead_mean": float(np.mean(v["lead"])) if v["lead"] else None} for k, v in by_kind.items()},
        "lead_by_priority": {k: float(np.mean(v)) for k, v in sorted(by_prio.items())},
    })
    out["order"] = o

    # ───────────────────────────── task and scheduling level
    first_ready, first_assigned, completed, deps = {}, {}, {}, {}
    reassign = suspend = fails = 0
    for e in ev:
        if e.type != "order_task":
            continue
        d = e.data
        tid = d["task_id"]
        if d["to"] == "ready":
            first_ready.setdefault(tid, e.tick)
            if d.get("frm") == "assigned":
                reassign += 1
        elif d["to"] == "assigned":
            first_assigned.setdefault(tid, e.tick)
        elif d["to"] == "suspended":
            suspend += 1
        elif d["to"] == "failed":
            fails += 1
        elif d["to"] == "completed":
            completed[tid] = e.tick
    tables = trace.get("tables") or {}
    for row in tables.get("tasks", []):
        if row.get("depends_on"):
            deps[row["task_id"]] = row["depends_on"].split("|")
    latency = [first_assigned[k] - first_ready[k] for k in first_assigned if k in first_ready]
    med = float(np.median(latency)) if latency else None
    starving = sum(1 for k, r in first_ready.items()
                   if med is not None and (first_assigned.get(k, T) - r) > 5 * max(med, 1))
    dep_delay = [first_ready[k] - max(completed[d] for d in ds) for k, ds in deps.items()
                 if k in first_ready and all(d in completed for d in ds)]
    rejections: Dict[str, int] = {}
    decisions = feasible_unassigned = eligible_ticks = 0
    for e in ev:
        if e.type == "schedule_decision":
            decisions += 1
            for row in e.data["tasks"]:
                eligible_ticks += 1
                rejections[row[5]] = rejections.get(row[5], 0) + 1
                feasible_unassigned += row[3] > 0 and row[5] != "assigned"
    rt = trace.get("runtime") or []
    out["scheduling"] = {
        "assignment_latency": _stats(latency), "reassignments": reassign, "suspensions": suspend,
        "attempt_failures": fails, "retries": o["retries"], "stalls_suspected": o["stalls"],
        "dependency_delay": _stats(dep_delay), "starving_tasks": starving,
        "decisions": decisions, "decision_outcomes": dict(sorted(rejections.items())),
        "feasible_unassigned_share": (feasible_unassigned / eligible_ticks) if eligible_ticks else None,
        "runtime_ms": _stats([r[1] for r in rt]), "candidates_per_tick": _stats([r[2] for r in rt]),
    }

    # ───────────────────────────── robot level
    act = np.zeros((n_rob, len(ACTIVITIES)), dtype=np.int64)
    dist = np.zeros(n_rob)
    loaded = np.zeros(n_rob)
    prev = trace.get("initial")
    for f in frames:
        for i, row in enumerate(f):
            act[i, row[6]] += 1
            if prev is not None:
                step = abs(row[0] - prev[i][0]) + abs(row[1] - prev[i][1])
                dist[i] += step
                if ACTIVITIES[row[6]] == "travel_loaded":
                    loaded[i] += step
        prev = f
    total = act.sum(axis=1).clip(min=1)
    A = {a: k for k, a in enumerate(ACTIVITIES)}
    busy = act[:, [A["travel_empty"], A["travel_loaded"], A["working"], A["waiting_partner"], A["waiting_blocked"]]].sum(1)
    sessions = sum(1 for e in ev if e.type == "charge_started")
    robots = []
    for i, r in enumerate(meta["robots"]):
        robots.append({"robot": r["name"], "type": r["type"], "distance": int(dist[i]), "loaded_distance": int(loaded[i]),
                       "empty_distance": int(dist[i] - loaded[i]), "utilization": float(busy[i] / total[i]),
                       **{"share_" + a: float(act[i, k] / total[i]) for k, a in enumerate(ACTIVITIES)}})
    energy = trace.get("energy_wh") or {}
    energy_wh = float(sum(energy.values())) if energy else None
    for row in robots:
        row["energy_wh"] = energy.get(row["robot"])
    out["robot"] = {
        "per_robot": robots, "distance_total": int(dist.sum()), "loaded_distance_total": int(loaded.sum()),
        "utilization_mean": float((busy / total).mean()), "utilization_jain": _jain(busy),
        "workload_cv": float(busy.std() / busy.mean()) if busy.mean() > 0 else None,
        "activity_share": {a: float(act[:, k].sum() / max(1, act.sum())) for k, a in enumerate(ACTIVITIES)},
        "charge_sessions": sessions, "charging_ticks": int(act[:, A["charging"]].sum()),
        "dock_queue_ticks": int(act[:, A["dock_queue"]].sum()),
        "min_soc": min((row[2] for f in frames for row in f), default=None),
        "stranded": sum(1 for e in ev if e.type == "robot_stranded"),
        "breakdowns": sum(1 for e in ev if e.type == "robot_failed"),
    }

    # ───────────────────────────── coalition level
    first_join, first_start, members, carriers = {}, {}, {}, {}
    for e in ev:
        if e.type == "robot_joined":
            first_join.setdefault(e.task, e.tick)
            members.setdefault(e.task, set()).update(e.robots)
            if e.data.get("role") in ("carrier", "carriers"):
                carriers.setdefault(e.task, set()).update(e.robots)
        elif e.type == "stage_started":
            first_start.setdefault(e.task, e.tick)
    formation = [first_start[k] - first_join[k] for k in first_start if k in first_join]
    committed = int(busy.sum())
    waitp = int(act[:, A["waiting_partner"]].sum())
    out["coalition"] = {
        "formation_time": _stats(formation),
        "waiting_ratio": (waitp / committed) if committed else None,
        "partner_waiting_ticks": waitp,
        "size": _stats([len(v) for v in members.values()]),
        "joint_transport_tasks": sum(1 for v in carriers.values() if len(v) > 1),
        "coalitions_formed": len(first_start),
    }

    # ───────────────────────────── system level
    shipped = sum(1 for e in ev if e.type == "inventory" and e.data.get("op") == "package_shipped")
    stored_in = sum(1 for e in ev if e.type == "inventory" and e.data.get("op") == "package_put")
    blocked_moves = sum(1 for e in ev if e.type == "collision_blocked")
    robot_ticks = max(1, int(act.sum()))
    out["system"] = {
        "deliveries": shipped + stored_in, "packages_shipped": shipped, "packages_stored_inbound": stored_in,
        "tasks_completed": o["tasks_completed"],
        "energy_wh": energy_wh,
        "energy_wh_per_order": (energy_wh / o["completed"]) if energy_wh and o["completed"] else None,
        "energy_wh_per_delivery": (energy_wh / (shipped + stored_in)) if energy_wh and (shipped + stored_in) else None,
        "distance_per_delivery": (dist.sum() / (shipped + stored_in)) if (shipped + stored_in) else None,
        "congestion_blocked_moves_per_1000_robot_ticks": 1000.0 * blocked_moves / robot_ticks,
        "congestion_waiting_share": float(act[:, A["waiting_blocked"]].sum() / robot_ticks),
    }
    out["order"]["energy_wh_per_order"] = out["system"]["energy_wh_per_order"]

    # ───────────────────────────── bottlenecks and diagnosis
    out["bottlenecks"] = diagnose(trace, closed, arrival, T)
    out["primary"] = {
        "order_throughput_per_1000": o["throughput_per_1000"], "order_lead_time_mean": o["lead_time_mean"],
        "on_time_rate": o["on_time_rate"], "energy_wh_per_order": out["system"]["energy_wh_per_order"],
        "coalition_waiting_ratio": out["coalition"]["waiting_ratio"],
        "unfinished_orders": o["unfinished"], "unfinished_age_mean": o["unfinished_age_mean"],
    }
    return out


NON_PROGRESS = [c for c in CAUSES if c not in ("travel", "execution")]
CATEGORY = {"sequencing": "scheduling decision", "coalition_unavailable": "coalition infeasibility (robots busy)",
            "coalition_waiting": "coalition waiting (partner)", "capability": "insufficient capabilities",
            "battery": "battery constraints", "charging_contention": "charging contention",
            "congestion": "navigation congestion", "unreachable": "navigation (no route)",
            "blocked_dependency": "task dependencies", "blocked_inventory": "inventory condition",
            "blocked_access": "access (no loaded route)", "breakdown": "robot breakdown",
            "failure_recovery": "failure and retry", "unassigned": "no scheduler decision",
            "deadlock": "navigation deadlock (suspected stall)",
            "travel": "travel", "execution": "execution"}


def diagnose(trace: Dict, closed: Dict, arrival: Dict, T: int) -> Dict:
    spans = trace.get("spans") or {}
    tasks = {r["task_id"]: r for r in (trace.get("tables") or {}).get("tasks", [])}
    orders = {r["order_id"]: r for r in (trace.get("tables") or {}).get("orders", [])}
    total: Dict[str, int] = {}
    for sp in spans.values():
        for a, b, c in sp:
            total[c] = total.get(c, 0) + b - a + 1
    s = sum(total.values()) or 1
    unsuccessful = []
    for oid, row in orders.items():
        if row["state"] == "completed" or (row["state"] == "cancelled"
                                           and not str(row.get("reason", "")).startswith("rejected")):
            continue
        ticks: Dict[str, int] = {}
        evidence: Dict[str, List[int]] = {}
        failure = []
        for tid in (row.get("tasks") or "").split("|"):
            if not tid:
                continue
            for a, b, c in spans.get(tid, []):
                ticks[c] = ticks.get(c, 0) + b - a + 1
                if c not in evidence or (b - a) > evidence[c][1] - evidence[c][0]:
                    evidence[c] = [a, b, tid]
            tr = tasks.get(tid, {})
            if tr.get("failure_reason"):
                failure.append({"task": tid, "reason": tr["failure_reason"], "blocking_resource": tr.get("blocking_resource"),
                                "last_progress_tick": tr.get("last_progress_tick")})
        np_ticks = {c: v for c, v in ticks.items() if c in NON_PROGRESS}
        stalled = [tid for tid in (row.get("tasks") or "").split("|") if tid and (tasks.get(tid, {}).get("stalls") or 0) > 0]
        if row["state"] == "failed" and failure:
            r0 = failure[-1]["reason"]
            cause = ("congestion" if "waiting_cell" in r0 or "no_progress" in r0 else
                     "capability" if "payload" in r0 else "blocked_inventory" if r0.startswith("inventory") else
                     "failure_recovery")
        elif row["state"] == "cancelled":
            cause = "blocked_inventory"
        elif stalled:                                 # its crew stopped making progress (stall detector)
            cause = "deadlock"
            st = [(sp[0], sp[1], tid) for tid in stalled for sp in spans.get(tid, [])
                  if sp[2] in ("congestion", "coalition_waiting")]
            evidence["deadlock"] = list(max(st, key=lambda x: (x[1] - x[0], x[2]))) if st else None
        else:
            cause = max(np_ticks, key=lambda c: (np_ticks[c], c)) if np_ticks else "travel"
        unsuccessful.append({"order_id": oid, "state": row["state"], "age": T - row["arrival"], "cause": cause,
                             "category": CATEGORY.get(cause, cause), "ticks_by_cause": ticks,
                             "evidence_span": evidence.get(cause), "failures": failure, "reason": row.get("reason")})
    by_cat: Dict[str, int] = {}
    for u in unsuccessful:
        by_cat[u["category"]] = by_cat.get(u["category"], 0) + 1
    return {"task_ticks_by_cause": dict(sorted(total.items())),
            "task_share_by_cause": {c: v / s for c, v in sorted(total.items())},
            "non_progress_share": sum(v for c, v in total.items() if c in NON_PROGRESS) / s,
            "unsuccessful_orders": unsuccessful, "unsuccessful_by_category": by_cat}
