"""Order scenarios on the intermediate and advanced warehouses, compound (dependency) orders and
sequenced lines, consistency of the scheduling interfaces across warehouses."""
import os
import sys

import pytest

import battery_tarware as bt
from battery_tarware.config import ConfigError
from battery_tarware.orders.run import run_episode
from battery_tarware.orders.scenarios import order_scenario

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "orders_v2"))
import workloads  # noqa: E402

SRC = {"kind": "poisson", "rate": 0.05, "max_orders": 40,
       "mix": {"outbound": 0.3, "inbound": 0.3, "relocation": 0.1, "cross_dock": 0.15, "replenish": 0.15},
       "lines": {"kind": "choice", "values": [1, 2, 3]}, "sequence": 0.5,
       "due_after": {"kind": "uniform", "low": 400, "high": 900}}


@pytest.mark.parametrize("base", ["basic", "intermediate", "advanced"])
def test_order_scenarios_validate_and_run(base):
    cfg = order_scenario(source=SRC, base=base)
    bt.ScenarioConfig.from_dict(cfg)                      # task types fit the order operations
    res = run_episode(cfg, "nsp", 0, 700, 0)              # inventory invariants are checked every tick
    m = res["metrics"]
    assert m["order"]["orders_arrived"] > 10 and m["order"]["completed"] > 0
    types = {t.type.name for t in res["sim"].tasks.values()}
    if base == "intermediate":
        assert {"tote", "case"} <= types
    if base == "advanced":
        assert "checked" in types                          # AGV + manipulator + drone with a QC stage
    assert m["scheduling"]["stalls_suspected"] == 0


def test_compound_orders_and_sequences():
    cfg = order_scenario(source=dict(SRC, max_orders=30), base="basic")
    res = run_episode(cfg, "fifo", 3, 600, 0)
    mgr = res["sim"].orders
    seen = set()
    for o in mgr.orders.values():
        ts = [mgr.tasks[i] for i in o.task_ids]
        ops = [t.op for t in ts]
        if o.kind == "cross_dock":
            seen.add("cross_dock")
            for a, b in zip(ts[::2], ts[1::2]):
                assert (a.op, b.op) == ("inbound", "outbound") and a.task_id in b.depends_on
                assert a.package_id == b.package_id                       # the received package ships
        if o.kind == "replenish":
            seen.add("replenish")
            for a, b in zip(ts[::2], ts[1::2]):
                assert (a.op, b.op) == ("relocation", "outbound") and a.task_id in b.depends_on
                assert res["sim"].orders.inventory.packages[b.package_id].history[0]   # a real package
        if len(ts) > 1 and o.kind in ("outbound", "inbound", "relocation") and any(t.depends_on for t in ts):
            seen.add("sequence")
            assert all(ts[i - 1].task_id in ts[i].depends_on for i in range(1, len(ts)))
        assert ops
    assert {"cross_dock", "replenish", "sequence"} <= seen


def test_generated_arrivals_identical_across_rules():
    cfg = workloads.build("intermediate", "2.1", "standard", max_steps=600)
    arr = {}
    for rule in ("fifo", "nsp", "plb"):
        res = run_episode(cfg, rule, 5, 600, 0, record=False)
        arr[rule] = [(e.tick, e.data["kind"], e.data["priority"], e.data["due"])
                     for e in res["sim"].events.events if e.type == "order_arrived"]
    assert arr["fifo"] == arr["nsp"] == arr["plb"] and len(arr["fifo"]) > 10


def test_unknown_compound_operation_rejected():
    cfg = order_scenario(source=dict(SRC, mix={"cross_dock": 1.0}), base="basic")
    del cfg["orders"]["operations"]["inbound"]
    with pytest.raises(ConfigError):
        bt.ScenarioConfig.from_dict(cfg)
        bt.make(cfg).reset(seed=0)


def test_busy_heavy_carriers_are_unavailable_not_unreachable():
    # intermediate: a 24 kg relocation only a pallet AGV (60 kg) can carry; tote AGVs (20 kg) are free
    cfg = workloads.build("intermediate", "2.1", "standard", max_steps=2000)
    res = run_episode(cfg, "fifo", 0, 2000, 0)
    out = res["metrics"]["scheduling"]["decision_outcomes"]
    assert out.get("infeasible:unreachable", 0) == 0, out
