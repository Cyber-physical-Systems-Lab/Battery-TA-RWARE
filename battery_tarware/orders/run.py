"""Run one recorded episode with an order scheduler: metrics, trace and ledgers.

    from battery_tarware.orders.run import run_episode
    res = run_episode(order_scenario(...), rule="plb", seed=0, max_steps=3000)
    res["metrics"]["primary"], res["trace"]          # trace -> dashboard / JSON
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import battery_tarware as bt
from battery_tarware.orders.metrics import evaluate
from battery_tarware.orders.reports import order_table, package_table, shelf_table, task_table
from battery_tarware.orders.scheduler import OrderScheduler
from battery_tarware.orders.trace import TraceRecorder


def tables(sim, tracker=None) -> Dict[str, Any]:
    m = sim.orders
    out = {"orders": order_table(m), "tasks": task_table(m), "packages": package_table(m), "shelves": shelf_table(m)}
    if tracker is not None:
        out["robots"] = bt.robot_table(tracker)
        out["trips"] = bt.package_table(tracker)
    return out


def run_episode(scenario, rule: str = "fifo", seed: int = 0, max_steps: Optional[int] = None, warmup: int = 0,
                scheduler_kwargs: Optional[Dict] = None, record: bool = True, coordinator=None) -> Dict[str, Any]:
    kw = {} if max_steps is None else {"max_steps": max_steps}
    env = bt.make(scenario, **kw)
    sim = env.unwrapped.sim
    co = coordinator or OrderScheduler(rule, **(scheduler_kwargs or {}))
    env.reset(seed=seed)
    co.reset(env)
    rec = TraceRecorder(sim, co if hasattr(co, "runtime") else None, label=rule)
    tracker = bt.PackageTracker(sim)
    t0 = time.perf_counter()
    while True:
        _, _, term, trunc, _ = env.step(co.act(env))
        rec.update()
        tracker.update()
        if term or trunc:
            break
    wall = time.perf_counter() - t0
    tab = tables(sim, tracker)
    trace = rec.to_dict(tables=tab)
    metrics = evaluate(trace, warmup)
    metrics["meta"]["wall_seconds"] = wall
    trace["metrics"] = metrics
    return {"metrics": metrics, "trace": trace if record else None, "sim": sim, "scheduler": co}
