"""Order-level metrics (from the event log only) and ledger tables (orders, tasks, packages, shelves).

``order_metrics`` reads nothing but events, so dashboards, CSVs and metrics share one source of
truth; the tests reconcile it with the manager's state.

Definitions (time in ticks; horizon ``T`` = the last tick of the run; window ``(warmup, T]``):

=====================  =====================================================================
orders_arrived         orders that arrived by ``T`` (including rejected ones)
completed / failed /   orders in that state at ``T``; ``unfinished`` = still open (censored: not
cancelled /            counted as failed, reported by count and age ``T - arrival``)
unfinished
throughput_per_1000    orders completed in the window / window length x 1000
lead_time_mean / p95   completion - arrival, completed orders only
on_time_rate           orders with a due date whose outcome is known by ``T`` (completed, failed,
                       or still open past their due date; cancelled ones excluded): share
                       completed at or before the due date. Open orders not yet due are
                       censored (``on_time_censored``).
energy_wh_per_order    total robot energy (if given) / completed orders
tasks_completed        order-layer tasks completed (an order completes only when all its tasks do)
retries / stalls       dispatch retries and suspected stalls
=====================  =====================================================================
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def order_metrics(events, horizon: int, warmup: int = 0, energy_kwh: Optional[float] = None) -> Dict:
    arrival: Dict[str, int] = {}
    due: Dict[str, Optional[int]] = {}
    closed: Dict[str, tuple] = {}
    tasks_completed = retries = stalls = 0
    for e in events:
        if e.tick > horizon:
            break
        if e.type == "order_arrived":
            arrival[e.data["order_id"]] = e.tick
            due[e.data["order_id"]] = e.data.get("due")
        elif e.type in ("order_completed", "order_failed", "order_cancelled"):
            closed[e.data["order_id"]] = (e.type[6:], e.tick)
        elif e.type == "order_task":
            if e.data["to"] == "completed":
                tasks_completed += 1
            elif e.data["to"] == "retrying":
                retries += 1
        elif e.type == "stall_suspected":
            stalls += 1
    state = {o: closed.get(o, ("unfinished", None))[0] for o in arrival}
    done = [o for o in arrival if state[o] == "completed"]
    lead = [closed[o][1] - arrival[o] for o in done]
    window = max(1, horizon - warmup)
    in_window = sum(1 for o in done if warmup < closed[o][1] <= horizon)
    known = hit = censored = 0
    for o, d in due.items():
        if d is None or state[o] == "cancelled":
            continue
        if state[o] == "unfinished" and d > horizon:
            censored += 1
            continue
        known += 1
        hit += state[o] == "completed" and closed[o][1] <= d
    ages = [horizon - arrival[o] for o in arrival if state[o] == "unfinished"]
    count = {k: sum(1 for o in arrival if state[o] == k) for k in ("completed", "failed", "cancelled", "unfinished")}
    return {
        "horizon": horizon, "warmup": warmup, "orders_arrived": len(arrival), **count,
        "throughput_per_1000": 1000.0 * in_window / window,
        "lead_time_mean": float(np.mean(lead)) if lead else None,
        "lead_time_p95": float(np.percentile(lead, 95)) if lead else None,
        "on_time_rate": hit / known if known else None, "on_time_known": known, "on_time_censored": censored,
        "energy_wh_per_order": (1000.0 * energy_kwh / len(done)) if (energy_kwh is not None and done) else None,
        "unfinished_age_mean": float(np.mean(ages)) if ages else None,
        "unfinished_age_max": max(ages) if ages else None,
        "tasks_completed": tasks_completed, "retries": retries, "stalls": stalls,
    }


def episode_order_metrics(sim, warmup: int = 0) -> Dict:
    return order_metrics(sim.events.events, sim.tick, warmup,
                         sum(r.energy_used_wh for r in sim.robots) / 1000.0)


# ──────────────────────────────────────────────────────────────── ledgers
def order_table(m) -> List[Dict]:
    return [{"order_id": o.order_id, "kind": o.kind, "source": o.source, "arrival": o.arrival,
             "priority": o.priority, "due": o.due, "state": o.state, "reason": o.reason, "closed": o.closed,
             "lead_time": None if o.state != "completed" else o.closed - o.arrival,
             "tasks": "|".join(o.task_ids), "groups": "|".join(o.group_ids)} for o in m.orders.values()]


def task_table(m) -> List[Dict]:
    out = []
    for t in m.tasks.values():
        last = t.failures[-1] if t.failures else None
        out.append({"task_id": t.task_id, "order_id": t.order_id, "group_id": t.group_id, "key": t.key, "op": t.op,
                    "state": t.state.value, "reason": t.reason, "final": t.final, "package_id": t.package_id,
                    "shelf_id": t.shelf_id, "station": t.station, "dest_location": t.dest_location,
                    "depends_on": "|".join(t.depends_on), "task_type": t.task_type, "weight_kg": t.weight_kg,
                    "attempt": t.attempt, "sim_tasks": "|".join(map(str, t.attempts)), "failures": len(t.failures),
                    "failure_reason": last.failure_reason if last else None,
                    "blocking_resource": last.blocking_resource if last else None,
                    "last_progress_tick": last.last_progress_tick if last else None, "stalls": t.stalls,
                    "created": t.created, "first_ready": t.first_ready, "completed_at": t.completed_at,
                    **{"wait_" + k: v for k, v in sorted(t.waits.items())}})
    return out


def package_table(m) -> List[Dict]:
    inv = m.inventory
    return [{"package_id": p.package_id, "status": p.status, "weight_kg": round(p.weight_kg, 3),
             "shelf_id": p.shelf_id, "rack_location_id": inv.rack_location_of(p), "cell": p.cell,
             "reserved_by": p.reserved_by, "created": p.created, "orders": "|".join(p.orders),
             "tasks": "|".join(p.tasks), "moves": len(p.history),
             "last": p.history[-1][2] if p.history else None} for p in inv.packages.values()]


def shelf_table(m) -> List[Dict]:
    return [{"shelf_id": s.shelf_id, "status": s.status, "location_id": s.location_id, "home_id": s.home_id,
             "cell": s.cell, "packages": "|".join(s.packages), "incoming": "|".join(s.incoming),
             "busy_task": s.busy_task} for s in m.inventory.shelves.values()]


def replay_inventory(events) -> Dict[str, Dict]:
    """Rebuild every package's final record (status, shelf, rack location, reserved_by) from the event
    log alone (the same replay the dashboard does per tick). Events are rows ``[tick, type, robots,
    task, data]`` or ``Event`` objects."""
    pk: Dict[str, Dict] = {}
    on: Dict[str, List[str]] = {}
    shelf: Dict[str, Dict] = {}
    for e in events:
        typ, d = (e[1], e[4]) if isinstance(e, (list, tuple)) else (e.type, e.data)
        if typ != "inventory":
            continue
        op = d["op"]
        if op == "initialised":
            for s, (loc, pids) in d["shelves"].items():
                on[s] = list(pids)
                shelf[s] = {"status": "stored", "loc": loc}
                for p in pids:
                    pk[p] = {"status": "stored", "shelf": s, "loc": loc, "reserved": None}
        elif op == "package_arrived":
            pk[d["package_id"]] = {"status": "arrived", "shelf": None, "loc": None, "reserved": None}
        elif op in ("package_reserved", "slot_reserved"):
            pk[d["package_id"]]["reserved"] = d["task_id"]
        elif op == "package_unreserved":
            pk[d["package_id"]]["reserved"] = None
        elif op in ("shelf_lifted", "shelf_dropped", "shelf_reloaded", "shelf_placed"):
            st = {"shelf_lifted": "in_transit", "shelf_dropped": "dropped", "shelf_reloaded": "in_transit",
                  "shelf_placed": "stored"}[op]
            loc = d.get("location_id") if op == "shelf_placed" else (None if op == "shelf_lifted" else shelf[d["shelf_id"]]["loc"])
            shelf[d["shelf_id"]] = {"status": st, "loc": loc}
            for p in on.get(d["shelf_id"], []):
                pk[p].update(status=st, loc=loc)
        elif op == "package_put":
            on.setdefault(d["shelf_id"], []).append(d["package_id"])
            pk[d["package_id"]].update(status=shelf[d["shelf_id"]]["status"], shelf=d["shelf_id"],
                                       loc=shelf[d["shelf_id"]]["loc"], reserved=None)
        elif op == "package_shipped":
            on[d["shelf_id"]].remove(d["package_id"])
            pk[d["package_id"]].update(status="shipped", shelf=None, loc=None, reserved=None)
        elif op == "package_cancelled":
            pk[d["package_id"]].update(status="cancelled", reserved=None)
        elif op == "slot_released":
            pk[d["package_id"]]["reserved"] = None
    return pk
