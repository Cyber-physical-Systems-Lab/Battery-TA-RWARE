"""Order layer (Phase 1): inventory, dependencies, extended lifecycle, failures and retries,
cancellation, order completion and order metrics, reconciled with the event log."""
import copy

import pytest

import battery_tarware as bt
from battery_tarware.config import ConfigError
from battery_tarware.orders import InventoryError, OTaskState
from battery_tarware.orders.monitor import ProgressMonitor
from battery_tarware.orders.reports import episode_order_metrics, order_table, package_table, task_table
from battery_tarware.orders.scenarios import order_scenario
from battery_tarware.robots import RobotStatus

# basic map: L0000 (3,2) L0001 (3,3) L0002 (3,6) ... L0010 (4,10); goals (15,3), (15,10)
INITIAL = {"L0000": 1, "L0001": 1, "L0002": 1, "L0010": 0, "L0020": 1}


def make(trace=(), initial=INITIAL, lifecycle=None, weight=4.0, source=None, max_steps=4000, seed=0):
    src = source or {"kind": "trace", "orders": list(trace)}
    inv = {"package_weight": weight} if initial is None else {"initial": initial, "package_weight": weight}
    env = bt.make(order_scenario(source=src, inventory=inv, lifecycle=lifecycle or {}), max_steps=max_steps)
    env.reset(seed=seed)
    return env, env.unwrapped.sim


def run(env, ticks=None, until=None, co=None):
    co = co or bt.GreedyCoalitionCoordinator("fifo")
    sim = env.unwrapped.sim
    for _ in range(ticks or 10 ** 6):
        _, _, term, trunc, _ = env.step(co.act(env))
        if term or trunc or (until is not None and until(sim)):
            break
    return sim


def events(sim, type_, **match):
    return [e for e in sim.events.events if e.type == type_ and all(e.data.get(k) == v for k, v in match.items())]


def break_robot(sim, rid, ticks):
    r = sim.robots[rid]
    r.resume_status = RobotStatus.TASK if r.task_id is not None else RobotStatus.IDLE
    r.status, r.repair_left = RobotStatus.FAILED, ticks
    r.clear_path()


# ─────────────────────────────────────────────────────────────── compatibility
def test_task_only_scenarios_have_no_order_layer():
    for name in bt.SCENARIOS:
        env = bt.make(name)
        assert env.unwrapped.sim.orders is None
        assert env.unwrapped.config.orders is None


def test_config_validation():
    cfg = order_scenario()
    bt.ScenarioConfig.from_dict(cfg)
    rt = bt.ScenarioConfig.from_dict(bt.ScenarioConfig.from_dict(cfg).to_dict())
    assert rt.orders == bt.ScenarioConfig.from_dict(cfg).orders
    bad = copy.deepcopy(cfg)
    bad["demand"] = bt.get_scenario("basic")["demand"]
    with pytest.raises(ConfigError, match="demand streams"):
        bt.ScenarioConfig.from_dict(bad)
    bad = copy.deepcopy(cfg)
    bad["orders"]["operations"]["relocation"] = {"task_types": ["standard"]}     # ends at its origin
    with pytest.raises(ConfigError, match="must end at its destination"):
        bt.ScenarioConfig.from_dict(bad)
    bad = copy.deepcopy(cfg)
    bad["orders"]["lifecycle"] = {"stall_tick": 5}
    with pytest.raises(ConfigError, match="unknown key"):
        bt.ScenarioConfig.from_dict(bad)
    bad = copy.deepcopy(cfg)
    bad["orders"]["source"]["mix"] = {"picking": 1.0}
    with pytest.raises(ConfigError, match="not configured"):
        bt.ScenarioConfig.from_dict(bad)


# ─────────────────────────────────────────────────────────────── inventory
def test_initial_inventory_and_ids():
    env, sim = make()
    inv = sim.orders.inventory
    assert [s.location_id for s in inv.shelves.values()] == ["L0000", "L0001", "L0002", "L0010", "L0020"]
    assert inv.shelves["S0003"].packages == []
    assert list(inv.packages) == ["P00000", "P00001", "P00002", "P00003"]
    assert inv.packages["P00003"].shelf_id == "S0004" and inv.rack_location_of(inv.packages["P00003"]) == "L0020"
    inv.check()
    ev = events(sim, "inventory", op="initialised")
    assert len(ev) == 1 and ev[0].data["shelves"]["S0004"] == ["L0020", ["P00003"]]


def test_package_id_persists_across_relocation_and_outbound():
    trace = [{"arrival": 0, "kind": "outbound", "tasks": [
        {"key": "move", "op": "relocation", "shelf": "S0000", "to": "L0004"},
        {"key": "ship", "op": "outbound", "package": "P00000", "depends_on": ["move"]}]}]
    env, sim = make(trace)
    run(env)
    m = sim.orders
    p = m.inventory.packages["P00000"]
    assert p.status == "shipped" and p.orders == ["O00000"] and p.tasks == ["T00001"]
    statuses = [h[1] for h in p.history]
    assert statuses == ["stored", "in_transit", "stored", "in_transit", "shipped"]
    assert m.inventory.shelves["S0000"].location_id == "L0004"            # shelf returned to its new home
    assert m.inventory.locations["L0000"].shelf_id is None
    assert m.orders["O00000"].state == "completed"


def test_inventory_conservation_under_mixed_load():
    cfg_src = {"kind": "poisson", "rate": 0.05, "max_orders": 60,
               "mix": {"outbound": 0.5, "inbound": 0.3, "relocation": 0.2}, "lines": {"kind": "choice", "values": [1, 2]}}
    env, sim = make(initial=None, source=cfg_src, weight={"kind": "uniform", "low": 0.5, "high": 16})
    run(env, ticks=2500)                                     # check_inventory runs every tick
    inv = sim.orders.inventory
    c = inv.counts()
    assert sum(c.values()) == inv.created
    assert c.get("shipped", 0) == inv.shipped == sum(
        1 for t in sim.orders.tasks.values() if t.op == "outbound" and t.state == OTaskState.COMPLETED)
    assert sum(len(s.packages) for s in inv.shelves.values()) == sum(
        c.get(k, 0) for k in ("stored", "in_transit", "dropped"))
    assert all(sum(1 for s in inv.shelves.values() if s.location_id == l) <= 1 for l in inv.locations)


def test_inventory_check_detects_corruption():
    env, sim = make()
    inv = sim.orders.inventory
    inv.shelves["S0001"].packages.append("P00000")           # one package on two shelves
    with pytest.raises(InventoryError):
        inv.check()


def test_inbound_puts_package_on_reserved_slot():
    trace = [{"arrival": 0, "op": "inbound", "packages": [{"weight_kg": 3.0}]}]
    env, sim = make(trace)
    m = sim.orders
    t = m.tasks["T00000"]
    assert t.shelf_id == "S0003"                              # the only shelf with a free slot
    assert m.inventory.shelves["S0003"].incoming == ["P00004"]
    run(env)
    p = m.inventory.packages["P00004"]
    assert p.status == "stored" and p.shelf_id == "S0003"
    assert m.inventory.rack_location_of(p) == "L0010"
    assert [h[1] for h in p.history] == ["arrived", "in_transit", "stored"]


def test_inbound_blocks_without_free_slot_then_proceeds():
    trace = [{"arrival": 0, "op": "inbound", "count": 2},
             {"arrival": 0, "op": "outbound", "packages": ["P00001"]}]
    env, sim = make(trace)
    m = sim.orders
    assert m.tasks["T00001"].state == OTaskState.BLOCKED and m.tasks["T00001"].reason == "inventory:no_free_slot"
    run(env)
    assert all(o.state == "completed" for o in m.orders.values())
    second = m.tasks["T00001"]
    assert second.shelf_id == "S0001"                         # the slot P00001 freed when it shipped
    ship = events(sim, "inventory", op="package_shipped", package_id="P00001")[0].tick
    assert m.tasks["T00001"].first_ready >= ship


# ─────────────────────────────────────────────────────────────── dependencies
def test_dependency_cycle_rejected():
    env, sim = make()
    sim.orders.submit({"kind": "outbound", "tasks": [
        {"key": "a", "op": "relocation", "shelf": "S0000", "depends_on": ["b"]},
        {"key": "b", "op": "outbound", "package": "P00001", "depends_on": ["a"]}]})
    with pytest.raises(ConfigError, match="cycle"):
        env.step([0] * len(sim.robots))


def test_blocked_tasks_are_never_released():
    trace = [{"arrival": 0, "kind": "outbound", "tasks": [
        {"key": "a", "op": "relocation", "shelf": "S0001", "to": "L0005"},
        {"key": "b", "op": "relocation", "shelf": "S0002", "to": "L0006", "depends_on": ["a"]},
        {"key": "c", "op": "outbound", "package": "P00000", "depends_on": ["b"]}]}]
    env, sim = make(trace)
    m = sim.orders
    assert [m.tasks[t].state for t in ("T00000", "T00001", "T00002")] == \
        [OTaskState.READY, OTaskState.BLOCKED, OTaskState.BLOCKED]
    assert len(sim.tasks) == 1                                # blocked tasks do not exist in the simulator
    run(env)
    done = {t: m.tasks[t].completed_at for t in ("T00000", "T00001", "T00002")}
    for prev, nxt in (("T00000", "T00001"), ("T00001", "T00002")):
        released = sim.tasks[m.tasks[nxt].attempts[0]].released
        assert released >= done[prev]
        joins = [e.tick for e in sim.events.events if e.type == "robot_joined" and e.task == m.tasks[nxt].attempts[0]]
        assert min(joins) >= done[prev]


def test_order_completes_only_when_all_tasks_complete():
    trace = [{"arrival": 0, "kind": "outbound", "tasks": [
        {"key": "ship1", "op": "outbound", "package": "P00000"},
        {"key": "move", "op": "relocation", "shelf": "S0002", "to": "L0007"},
        {"key": "ship2", "op": "outbound", "package": "P00002", "depends_on": ["move"]}]}]
    env, sim = make(trace)
    m = sim.orders
    first = None

    def check(s):
        nonlocal first
        if first is None and m.inventory.packages["P00000"].status == "shipped":
            first = s.tick
            assert m.orders["O00000"].state == "open"           # one package at its goal is not the order
        return False
    run(env, until=check)
    o = m.orders["O00000"]
    assert first is not None and o.state == "completed"
    assert o.closed == max(m.tasks[t].completed_at for t in o.task_ids)
    assert len(events(sim, "order_completed", order_id="O00000")) == 1


# ─────────────────────────────────────────────────────────────── feasibility
def test_assignments_respect_capabilities_and_payload():
    src = {"kind": "poisson", "rate": 0.05, "max_orders": 50, "mix": {"outbound": 0.6, "inbound": 0.4}}
    env, sim = make(initial=None, source=src, weight={"kind": "uniform", "low": 0.5, "high": 20})
    run(env, ticks=2500)
    for e in sim.events.events:
        if e.type == "robot_joined":
            t = sim.tasks[e.task]
            assert t.type.role(e.data["role"]).accepts(sim.robots[e.robots[0]])
        if e.type == "payload_picked" and not e.data.get("reload"):
            t = sim.tasks[e.task]
            assert sum(sim.robots[i].spec.payload_kg for i in e.robots) + 1e-9 >= t.weight_kg


def test_overweight_inbound_stays_blocked():
    env, sim = make([{"arrival": 0, "op": "inbound", "packages": [{"weight_kg": 80.0}]}], max_steps=300)
    run(env)
    t = sim.orders.tasks["T00000"]
    assert t.state == OTaskState.BLOCKED and t.reason.startswith("infeasible:payload") and not t.attempts


# ─────────────────────────────────────────────────────────────── failures, retries, waiting
def _stall_setup(lifecycle, broken_for):
    """Outbound of P00000; a broken-down robot stands on its rack cell (3, 2)."""
    env, sim = make([{"arrival": 0, "op": "outbound", "packages": ["P00000"]}], lifecycle=lifecycle, max_steps=1500)
    blocker = sim.robots[8]                                   # a picking robot (idle; never needed for solo)
    blocker.cell = (3, 2)
    break_robot(sim, blocker.id, broken_for)
    return env, sim


def test_stall_is_diagnostic_by_default():
    env, sim = _stall_setup({"stall_ticks": 40}, broken_for=200)
    run(env)
    st = events(sim, "stall_suspected")
    assert len(st) == 1 and st[0].data["action"] == "diagnostic_only"
    assert st[0].data["wait_reason"] == "waiting_cell" and "robot 8" in st[0].data["blocking_resource"]
    assert sim.orders.tasks["T00000"].attempt == 1 and sim.orders.orders["O00000"].state == "completed"


def test_stall_retry_then_recovery():
    env, sim = _stall_setup({"stall_ticks": 40, "auto_retry": True, "retry_delay": 5}, broken_for=60)
    run(env)
    t = sim.orders.tasks["T00000"]
    assert t.state == OTaskState.COMPLETED and t.attempt == 2 and len(t.failures) == 1
    f = t.failures[0]
    assert f.failure_reason.startswith("stall: no progress for 40 ticks (waiting_cell)")
    assert "robot 8" in f.blocking_resource and f.attempt == 1 and f.last_progress_tick < f.tick
    assert f.snapshot["task"]["id"] == t.attempts[0] and f.snapshot["crew"]
    states = [h[2] for h in t.history]
    assert states[:3] == ["pending", "ready", "assigned"] and "failed" in states and "retrying" in states
    assert sim.tasks[t.attempts[0]].state.value == "cancelled"
    assert sim.orders.orders["O00000"].state == "completed"


def test_retry_budget_exhausted_fails_order_and_releases_inventory():
    env, sim = _stall_setup({"stall_ticks": 30, "auto_retry": True, "retry_delay": 5, "max_dispatch_retries": 2},
                            broken_for=10 ** 6)
    run(env)
    m = sim.orders
    t, o = m.tasks["T00000"], m.orders["O00000"]
    assert t.state == OTaskState.FAILED and t.final and t.attempt == 3 and len(t.failures) == 3
    assert o.state == "failed" and "after 3 attempt(s)" in o.reason
    assert m.inventory.packages["P00000"].reserved_by is None
    assert m.inventory.shelves["S0000"].busy_task is None and m.inventory.shelves["S0000"].location_id == "L0000"
    assert episode_order_metrics(sim)["failed"] == 1


def test_failures_are_reproducible():
    def failures():
        env, sim = _stall_setup({"stall_ticks": 30, "auto_retry": True, "retry_delay": 5, "max_dispatch_retries": 1},
                                broken_for=10 ** 6)
        run(env)
        return [(f.tick, f.attempt, f.sim_task, f.failure_reason, f.blocking_resource, f.last_progress_tick)
                for f in sim.orders.tasks["T00000"].failures]
    assert failures() == failures()


def test_waiting_for_partner_is_not_a_stall():
    """A standard (picker) task whose pickers break down after pick-up: the carrier waits for a picker
    to return the shelf, far longer than the stall threshold, without a suspected stall."""
    env, sim = make([{"arrival": 0, "op": "outbound", "packages": ["P00000"]}], weight=8.0,
                    lifecycle={"stall_ticks": 30, "auto_retry": True})
    m = sim.orders
    run(env, until=lambda s: m.inventory.packages["P00000"].status == "shipped")
    for r in sim.robots:
        if r.spec.name == "picking_robot":
            break_robot(sim, r.id, 150)
    run(env)
    assert not events(sim, "stall_suspected")
    t = m.tasks["T00000"]
    assert t.state == OTaskState.COMPLETED and t.attempt == 1
    assert t.waits.get("waiting_partner", 0) > 30


def test_explicit_failure_report_and_retry():
    env, sim = make([{"arrival": 0, "op": "outbound", "packages": ["P00000"]}], lifecycle={"retry_delay": 3})
    m = sim.orders
    ok, why = m.fail_task("T00000", "path planning failed 3 times", blocking_resource="cell (5, 2)")
    assert ok, why
    t = m.tasks["T00000"]
    assert t.state == OTaskState.RETRYING and t.failures[0].blocking_resource == "cell (5, 2)"
    run(env)
    assert t.state == OTaskState.COMPLETED and t.attempt == 2


def test_failure_refused_after_pickup():
    env, sim = make([{"arrival": 0, "op": "outbound", "packages": ["P00000"]}])
    m = sim.orders
    run(env, until=lambda s: m.inventory.packages["P00000"].status == "in_transit")
    ok, why = m.fail_task("T00000", "test")
    assert not ok and "left its origin" in why


def test_monitor_threshold_on_baseline_traces():
    """The default 100-tick threshold flags nothing in successful task-only greedy runs."""
    for name in ("basic", "intermediate"):
        env = bt.make(name, max_steps=1500)
        sim = env.unwrapped.sim
        env.reset(seed=0)
        mon = ProgressMonitor(sim, 100)
        co = bt.GreedyCoalitionCoordinator("fifo")
        for _ in range(1500):
            env.step(co.act(env))
            mon.update()
        assert mon.reports == [], (name, [(r.sim_task, r.wait_reason, r.blocking_resource) for r in mon.reports])


# ─────────────────────────────────────────────────────────────── cancellation
def test_cancel_order_releases_reservations():
    trace = [{"arrival": 0, "kind": "mixed", "tasks": [
        {"key": "in", "op": "inbound", "weight_kg": 2.0},
        {"key": "mv", "op": "relocation", "shelf": "S0001", "to": "L0008"},
        {"key": "out", "op": "outbound", "package": "P00002", "depends_on": ["mv"]}]}]
    env, sim = make(trace)
    m = sim.orders
    left = m.cancel_order("O00000", "customer withdrew")
    assert left == []
    inv = m.inventory
    assert all(m.tasks[t].state == OTaskState.CANCELLED for t in m.orders["O00000"].task_ids)
    assert inv.packages["P00004"].status == "cancelled" and inv.shelves["S0003"].incoming == []
    assert inv.locations["L0008"].reserved_by is None and inv.packages["P00002"].reserved_by is None
    assert all(sim.tasks[i].state.value == "cancelled" for i in sim.tasks)
    inv.check()
    run(env, ticks=50)
    assert all(not r.task_id for r in sim.robots)


def test_cancel_after_pickup_finishes_physically():
    env, sim = make([{"arrival": 0, "op": "relocation", "shelves": ["S0000"], "to": ["L0003"]}])
    m = sim.orders
    run(env, until=lambda s: m.inventory.shelves["S0000"].status == "in_transit")
    assert m.cancel_order("O00000", "no longer needed") == ["T00000"]
    run(env)
    assert m.orders["O00000"].state == "cancelled"
    assert m.tasks["T00000"].state == OTaskState.COMPLETED
    assert m.inventory.shelves["S0000"].location_id == "L0003"


# ─────────────────────────────────────────────────────────────── metrics and determinism
def test_order_metrics_reconcile_with_state_and_events():
    src = {"kind": "poisson", "rate": 0.25, "max_orders": 400,
           "mix": {"outbound": 0.6, "inbound": 0.25, "relocation": 0.15}, "due_after": {"kind": "fixed", "value": 250}}
    env, sim = make(initial=None, source=src, weight={"kind": "uniform", "low": 0.5, "high": 16}, max_steps=1200)
    run(env)
    m = sim.orders
    met = episode_order_metrics(sim)
    states = {}
    for o in m.orders.values():
        states[o.state] = states.get(o.state, 0) + 1
    assert met["orders_arrived"] == len(m.orders)
    assert met["completed"] == states.get("completed", 0)
    assert met["unfinished"] == states.get("open", 0) > 0         # truncated: censored, not failed
    assert met["failed"] == states.get("failed", 0) == 0
    lt = [r["lead_time"] for r in order_table(m) if r["state"] == "completed"]
    assert met["lead_time_mean"] == pytest.approx(sum(lt) / len(lt))
    assert met["tasks_completed"] == sum(1 for t in m.tasks.values() if t.state == OTaskState.COMPLETED)
    for o in m.orders.values():
        if o.state == "completed":
            assert o.closed == max(m.tasks[t].completed_at for t in o.task_ids)
    shipped = len(events(sim, "inventory", op="package_shipped"))
    assert shipped == m.inventory.shipped == sum(1 for r in package_table(m) if r["status"] == "shipped")
    assert len(task_table(m)) == len(m.tasks)
    assert met["energy_wh_per_order"] > 0 and 0 <= met["on_time_rate"] <= 1


def test_same_seed_same_order_history():
    def log(seed):
        src = {"kind": "poisson", "rate": 0.05, "max_orders": 30, "mix": {"outbound": 0.6, "inbound": 0.2,
                                                                          "relocation": 0.2}}
        env, sim = make(initial=None, source=src, weight={"kind": "uniform", "low": 0.5, "high": 16}, seed=seed)
        run(env, ticks=1500)
        return [(e.tick, e.type, e.task, sorted((k, str(v)) for k, v in e.data.items())) for e in sim.events.events]
    a = log(3)
    assert a == log(3) and a != log(4)


def test_reset_restarts_order_layer():
    env, sim = make([{"arrival": 0, "op": "outbound", "packages": ["P00000"]}])
    run(env)
    assert sim.orders.orders["O00000"].state == "completed"
    env.reset(seed=0)
    assert sim.orders.orders["O00000"].state == "open" and sim.orders.inventory.packages["P00000"].status == "stored"


def test_tracker_in_order_mode_links_trips_to_orders():
    trace = [{"arrival": 0, "kind": "outbound", "tasks": [
        {"key": "move", "op": "relocation", "shelf": "S0000", "to": "L0004"},
        {"key": "ship", "op": "outbound", "package": "P00000", "depends_on": ["move"]}]}]
    env, sim = make(trace)
    tr = bt.PackageTracker(sim)
    co = bt.GreedyCoalitionCoordinator("fifo")
    while True:
        _, _, term, trunc, _ = env.step(co.act(env))
        tr.update()
        if term or trunc:
            break
    rows = bt.package_table(tr)
    assert [r["package_id"] for r in rows] == ["M00000", "M00001"]
    assert [(r["order_task"], r["op"], r["shelf_id"], r["inventory_package"]) for r in rows] == \
        [("T00000", "relocation", "S0000", None), ("T00001", "outbound", "S0000", "P00000")]
    assert [r["status"] for r in rows] == ["delivered", "stored"]      # a relocation ends at its destination


def test_relocation_of_a_shelf_that_is_away_waits_without_error():
    """Regression: a relocation without a destination, accepted while its shelf is out on another task,
    must wait (blocked) instead of choosing a location from an unknown shelf cell."""
    env, sim = make([{"arrival": 0, "op": "outbound", "packages": ["P00000"]}])
    m = sim.orders
    run(env, until=lambda s: m.inventory.shelves["S0000"].status == "in_transit")
    m.submit({"op": "relocation", "shelves": ["S0000"]})
    run(env, ticks=1)
    t = m.tasks["T00001"]
    assert t.state == OTaskState.BLOCKED and t.reason == "inventory:shelf_in_transit" and t.dest_location is None
    run(env)
    assert t.state == OTaskState.COMPLETED and m.inventory.shelves["S0000"].location_id == t.dest_location
