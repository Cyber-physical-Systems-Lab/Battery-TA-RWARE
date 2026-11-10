"""Model 2.2: the helper work-cell reservation (docs/semantics.md, "Model 2.2").

The known model 2.1 gridlock (base order scenario, PLB, seed 1017) must still reproduce under 2.1
(2.1 is frozen) and must not occur under 2.2; model 2.1 trajectories stay identical to the
reference recorded before 2.2 existed; 2.2 helpers never target another task's work cell."""
import os
import sys

import pytest

import battery_tarware as bt
from battery_tarware.config import ConfigError

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments", "orders"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "checks"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "model_2_2"))


def _gridlock_run(model, ticks=600):
    import factors
    from battery_tarware.orders.run import run_episode
    cfg = factors.build()
    cfg["model"] = model
    return run_episode(cfg, "plb", 1017, ticks, 0, record=False)


def test_model_versions():
    assert bt.ScenarioConfig.from_dict(bt.get_scenario("basic")).model == "2.1"          # default unchanged
    assert bt.ScenarioConfig.from_dict(bt.get_scenario("basic", model="2.2")).model == "2.2"
    cfg = bt.get_scenario("intermediate", model="2.2", substitution=True)
    assert cfg["model"] == "2.2"
    bad = bt.get_scenario("basic")
    bad["model"] = "2.4"
    with pytest.raises(ConfigError):
        bt.ScenarioConfig.from_dict(bad)


def test_known_gridlock_reproduces_under_2_1():
    res = _gridlock_run("2.1")
    stalls = [e for e in res["sim"].events.events if e.type == "stall_suspected"]
    assert stalls, "model 2.1 is frozen: the documented gridlock must still reproduce"
    crew = {c["name"]: tuple(c["cell"]) for c in stalls[0].data["snapshot"]["crew"]}
    assert crew.get("picking_robot_1") == (11, 10)        # the helper on another task's work cell


def test_known_gridlock_absent_under_2_2():
    res = _gridlock_run("2.2", ticks=1500)
    assert not [e for e in res["sim"].events.events if e.type == "stall_suspected"]
    assert res["metrics"]["order"]["completed"] > 60


def test_helpers_never_target_foreign_work_cells_and_no_conflicts():
    import factors
    import gate
    from battery_tarware.orders.scheduler import OrderScheduler
    for model, expect_h1 in (("2.1", True), ("2.2", False)):
        cfg = factors.build()
        cfg["model"] = model
        env = bt.make(cfg, max_steps=600)
        sim = env.unwrapped.sim
        mon = gate.Monitor(sim)
        co = OrderScheduler("plb")
        env.reset(seed=1017)
        co.reset(env)
        while sim.tick < 600:
            prev = {r.id: r.cell for r in sim.robots}
            env.step(co.act(env))
            mon.after_tick(prev)
        mon.close()
        assert bool(mon.v.get("H1a")) == expect_h1, (model, mon.v.get("H1a", [])[:3])
        assert not mon.v.get("H1b") or model == "2.1"
        for k in ("S1", "S2", "S3", "L2"):
            assert not mon.v.get(k), (model, k, mon.v[k][:3])


def test_helper_avoids_foreign_cell_unit():
    """A 2.2 helper beside its stage cell steps off a cell that another staffed task needs."""
    import factors
    from battery_tarware.orders.scheduler import OrderScheduler
    cfg = factors.build()
    cfg["model"] = "2.2"
    env = bt.make(cfg, max_steps=200)
    sim = env.unwrapped.sim
    env.reset(seed=1017)
    co = OrderScheduler("plb")
    co.reset(env)
    checked = 0
    while sim.tick < 200:
        env.step(co.act(env))
        for r in sim.robots:
            if r.task_id is None:
                continue
            t = sim.tasks[r.task_id]
            if r.role == t.type.carrier_role or sim._option(r, t).presence != "adjacent":
                continue
            tgt = sim._target(r)
            foreign = sim._foreign_work_cells(t) - {t.stage_cell}
            if tgt is not None and tgt != r.cell:
                checked += 1
                assert tgt not in foreign or sim.tick == 0
    assert checked > 0


def test_model_2_0_and_2_1_trajectories_unchanged():
    import identity
    import json
    ref = json.load(open(os.path.join(ROOT, "experiments", "checks", "reference_identity.json")))
    now = identity.fingerprints()
    assert now == ref
