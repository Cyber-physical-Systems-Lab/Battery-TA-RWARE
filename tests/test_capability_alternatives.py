"""Role alternatives: one crew slot fillable through different capabilities, each with its own work
position (intermediate ``case`` support: picking robot on the shelf cell, or tote AGV with
``load_assist`` from the next cell)."""
import copy

import numpy as np
import pytest

import battery_tarware as bt
from battery_tarware import ScenarioConfig, Simulator
from battery_tarware.config import ConfigError, RoleSpec
from battery_tarware.robots import RobotStatus
from conftest import place, tiny_dict


def sub_sim(seed=0):
    sim = Simulator(ScenarioConfig.from_dict(bt.get_scenario("intermediate", substitution=True)))
    sim.reset(seed=seed)
    return sim


def by_type(sim, name):
    return [r for r in sim.robots if r.spec.name == name]


def new_case(sim):
    """An open case task at stage 0 with nobody assigned (released by the demand stream)."""
    for _ in range(400):
        for t in sim.open_tasks():
            if t.type.name == "case" and t.stage == 0 and not t.robots():
                return t
        sim.step([0] * len(sim.robots))
    raise AssertionError("no case task released")


# ───────────────────────────── configuration

def test_single_capability_roles_unchanged():
    rs = RoleSpec.from_dict({"name": "picker", "capability": "pick", "presence": "on_cell"}, "t")
    assert rs.capabilities == ("pick",) and rs.options[0].presence == "on_cell" and rs.capability_key == "pick"
    c = ScenarioConfig.from_dict(bt.get_scenario("intermediate"))
    assert ScenarioConfig.from_dict(c.to_dict()) == c


def test_alternatives_parse_and_round_trip():
    c = ScenarioConfig.from_dict(bt.get_scenario("intermediate", substitution=True))
    rs = next(t for t in c.task_types if t.name == "case").role("picker")
    assert rs.capability is None and rs.capabilities == ("pick", "load_assist")
    assert [(o.presence, o.radius) for o in rs.options] == [("on_cell", 1), ("adjacent", 1)]
    assert rs.capability_key == "pick|load_assist"
    assert ScenarioConfig.from_dict(c.to_dict()) == c
    assert "load_assist" in c.capabilities


@pytest.mark.parametrize("role,msg", [
    ({"name": "p", "capability": "pick", "alternatives": [{"capability": "x"}]}, "either capability or alternatives"),
    ({"name": "p", "alternatives": [{"capability": "a"}, {"capability": "a"}]}, "duplicate"),
    ({"name": "p", "alternatives": [{"capability": "a", "presence": "near"}]}, "presence"),
    ({"name": "p", "alternatives": [{"capability": "a", "colour": 1}]}, "unknown key"),
    ({"name": "p", "alternatives": [{"capability": "a|b"}]}, "without '|'"),
    ({"name": "p"}, "needs a capability"),
])
def test_bad_alternatives_rejected(role, msg):
    d = tiny_dict()
    d["task_types"][0]["roles"][1] = role
    with pytest.raises(ConfigError, match=msg):
        ScenarioConfig.from_dict(d)


def test_carrier_role_cannot_have_alternatives():
    d = tiny_dict()
    d["task_types"][0]["roles"][0] = {"name": "carrier", "alternatives": [{"capability": "transport", "presence": "on_cell"}]}
    with pytest.raises(ConfigError, match="carrier role cannot"):
        ScenarioConfig.from_dict(d)


def test_substitution_scope():
    with pytest.raises(ConfigError):
        bt.get_scenario("basic", substitution=True)
    with pytest.raises(ConfigError):
        bt.get_scenario("intermediate", model="2.0", substitution=True)
    base, sub = bt.get_scenario("intermediate"), bt.get_scenario("intermediate", substitution=True)
    caps = {r["name"]: r["capabilities"] for r in sub["robot_types"]}
    assert "load_assist" in caps["tote_agv"]
    assert all("load_assist" not in v for k, v in caps.items() if k != "tote_agv")
    for tb, ts in zip(base["task_types"], sub["task_types"]):
        if tb["name"] != "case":
            assert tb == ts                                  # pallet, oversize, tote unchanged
        else:
            assert tb["stages"] == ts["stages"] and tb["weight"] == ts["weight"]
    assert base["demand"] == sub["demand"] and base["fleet"] == sub["fleet"]


# ───────────────────────────── eligibility and joining

def test_who_may_fill_the_support_role():
    sim = sub_sim()
    t = new_case(sim)
    tote, pallet = by_type(sim, "tote_agv")[0], by_type(sim, "pallet_agv")[0]
    picker = by_type(sim, "picking_robot")[0]
    assert sim.can_join(tote, t, "picker") == (True, "picker")
    assert sim.can_join(picker, t, "picker") == (True, "picker")
    assert not sim.can_join(pallet, t, "picker")[0]          # transport alone is not load_assist


def test_tote_agv_cannot_support_a_pallet_task():
    sim = sub_sim()
    for _ in range(2000):
        ts = [t for t in sim.open_tasks() if t.type.name == "pallet" and t.stage == 0 and not t.robots()]
        if ts:
            break
        sim.step([0] * len(sim.robots))
    tote = by_type(sim, "tote_agv")[0]
    assert not sim.can_join(tote, ts[0], "picker")[0]


def test_carrier_and_support_are_distinct_robots():
    sim = sub_sim()
    t = new_case(sim)
    a = by_type(sim, "tote_agv")[0]
    assert sim.assign(t.id, a.id, "carrier")[0]
    ok, why = sim.can_join(a, t, "picker")
    assert not ok and "already" in why
    for c in sim.feasible_coalitions(t.id, limit=50):
        assert len(set(c.robots)) == len(c.robots)


def test_feasible_coalitions_offer_agv_support():
    sim = sub_sim()
    t = new_case(sim)
    sup = {sim.robots[c.assignment["picker"][0]].spec.name for c in sim.feasible_coalitions(t.id, limit=50)}
    assert "tote_agv" in sup and ({"picking_robot", "fast_picking_robot"} & sup)


# ───────────────────────────── work positions

def test_presence_follows_the_robots_alternative():
    sim = sub_sim()
    t = new_case(sim)
    cell = t.stage_cells[0]
    agv, picker = by_type(sim, "tote_agv")[1], by_type(sim, "picking_robot")[0]
    assert sim.assign(t.id, agv.id, "picker")[0]
    near = next(c for c in [(cell[0], cell[1] - 1), (cell[0], cell[1] + 1), (cell[0] - 1, cell[1]), (cell[0] + 1, cell[1])]
                if sim.layout.in_bounds(c))
    place(sim, agv.id, near)
    assert sim._present(agv, t, "picker", cell)               # adjacent counts for load_assist
    assert sim._target(agv) != cell                           # and it never aims at the shelf cell
    sim.release(agv.id)
    assert sim.assign(t.id, picker.id, "picker")[0]
    place(sim, picker.id, near)
    assert not sim._present(picker, t, "picker", cell)        # a picking robot must be on the cell


def test_observation_and_state_view_count_alternative_slots():
    sim = sub_sim()
    t = new_case(sim)
    caps = list(sim.capabilities)
    sv = bt.build_state_view(sim)
    tv = next(x for x in sv.tasks if x.id == t.id)
    assert ("picker", "pick|load_assist", 1) in tv.open_roles
    a = sv.to_arrays()
    i = [x.id for x in sv.tasks].index(t.id)
    assert a["task_open_capability"][i, caps.index("pick")] == 1
    assert a["task_open_capability"][i, caps.index("load_assist")] == 1
    assert t.required["pick"] == 1 and t.required["load_assist"] == 1


def test_agent_observation_counts_the_slot_under_both_capabilities():
    env = bt.make(bt.get_scenario("intermediate", substitution=True))
    env.reset(seed=0)
    sim = env.unwrapped.sim
    t = new_case(sim)
    ob = env.unwrapped.obs_builder
    obs = ob.build(sim.action_masks())
    k = sim.task_slots().index(t)
    row = obs[sim.robots[0].name]["tasks"][k]
    nc = len(ob.caps)
    open_caps = row[-2 * nc:-nc]
    assert open_caps[ob.caps.index("pick")] == 1 and open_caps[ob.caps.index("load_assist")] == 1


def test_action_mask_lets_a_tote_agv_take_the_case_support_slot():
    env = bt.make(bt.get_scenario("intermediate", substitution=True))
    env.reset(seed=0)
    sim = env.unwrapped.sim
    t = new_case(sim)
    slots = sim.task_slots()
    k = slots.index(t)
    tote, pallet = by_type(sim, "tote_agv")[0], by_type(sim, "pallet_agv")[0]
    # join actions are 1..K; the case slot is joinable for both tote AGVs (as carrier or support)
    assert sim.macro_action_mask(tote, slots)[1 + k] == 1
    sim2_mask = sim.macro_action_mask(pallet, slots)[1 + k]
    assert sim2_mask == 1                                     # a pallet AGV may still carry it


# ───────────────────────────── episodes

def test_greedy_episode_uses_agv_support_from_beside_the_shelf():
    env = bt.make(bt.get_scenario("intermediate", substitution=True), max_steps=1500)
    sim = env.unwrapped.sim
    co = bt.GreedyCoalitionCoordinator()
    env.reset(seed=1)
    agv_work = 0
    while True:
        _, _, te, tr, info = env.step(co.act(env))
        cells = [r.cell for r in sim.robots]
        for r in sim.robots:
            if r.task_id is None or r.role != "picker":
                continue
            t = sim.tasks[r.task_id]
            if t.type.name == "case" and r.can("load_assist") and r.worked:
                agv_work += 1
                assert r.cell != t.stage_cell
                assert sim.chebyshev(r.cell, t.stage_cell) <= 1
                assert cells.count(r.cell) == 1                # no cell shared with another robot
        if te or tr:
            break
    assert agv_work > 0 and info["metrics"]["stranded"] == 0


def test_without_substitution_trajectories_match_the_plain_scenario():
    def run(cfg):
        env = bt.make(cfg, max_steps=400)
        sim = env.unwrapped.sim
        co = bt.GreedyCoalitionCoordinator()
        env.reset(seed=2)
        out = []
        for _ in range(400):
            env.step(co.act(env))
            out.append(tuple((r.cell, r.task_id) for r in sim.robots))
        return out
    a = bt.get_scenario("intermediate")
    b = copy.deepcopy(a)
    assert run(a) == run(b)
    assert bt.get_scenario("intermediate", substitution=False) == a
