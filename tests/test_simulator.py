"""Engine mechanics on a tiny hand-made map (see conftest.TINY_MAP; two-way aisles).

Energy constants in the fixture are chosen so one tick (5 s) costs exactly:
idle 0.05 Wh, driving +0.10 Wh, payload +0.005 Wh/kg, handling +0.05 Wh; charging adds 1.0 Wh.
"""
import copy

import numpy as np
import pytest

from battery_tarware import ScenarioConfig, Simulator, build_state_view
from battery_tarware.robots import RobotStatus
from battery_tarware.tasks import ALLOWED_TRANSITIONS, TaskState
from conftest import AGV, PICKER, place, retrieval, run, tiny_dict

IDLE, MOVE, HANDLE, CHARGE = 0.05, 0.10, 0.05, 1.0


def sim_from(**over):
    return Simulator(ScenarioConfig.from_dict(tiny_dict(**over)))


def park_all(sim):
    """Put every robot on its own parking cell, so idle robots stay where they are."""
    used = set()
    for r in sim.robots:
        cells = [tuple(map(int, c)) for c in zip(*np.nonzero(sim._parking(r)))]
        c = next(c for c in cells if c not in used)
        used.add(c)
        place(sim, r.id, c)


def types(**changes):
    agv, pk = copy.deepcopy(AGV), copy.deepcopy(PICKER)
    agv.update(changes.get("agv", {}))
    pk.update(changes.get("picker", {}))
    return [agv, pk]


# ───────────────────────────── energy and charging

def test_idle_energy_is_exact(tiny):
    sim = tiny()
    park_all(sim)
    e0 = [r.energy_wh for r in sim.robots]
    run(sim, 10)
    for r, e in zip(sim.robots, e0):
        assert r.energy_wh == pytest.approx(e - 10 * IDLE)


def test_driving_and_charging_energy(tiny):
    sim = tiny(fleet=[{"type": "agv", "count": 1, "initial_soc": 0.5}, {"type": "picker", "count": 1}])
    park_all(sim)
    place(sim, 0, (2, 0))
    r = sim.robots[0]
    ok, _ = sim.send_to_charger(0, 0)                       # dock (0, 0), two cells away
    assert ok
    e0 = r.energy_wh
    run(sim, 2)
    assert r.cell == (0, 0) and r.distance == 2
    # tick 2 ends on the dock: charging starts the same tick
    assert r.energy_wh == pytest.approx(e0 - 2 * (IDLE + MOVE) + CHARGE)
    run(sim, 3)
    assert r.energy_wh == pytest.approx(e0 - 2 * (IDLE + MOVE) + 4 * CHARGE - 3 * IDLE)
    assert any(e.type == "charge_started" for e in sim.events.events)
    run(sim, 600)                                           # until full: stops and leaves the dock
    assert r.energy_wh == pytest.approx(r.spec.energy.capacity_wh, abs=1.0) or r.soc > 0.99
    assert any(e.type == "charge_stopped" and e.data["reason"] == "full" for e in sim.events.events)
    assert r.status == RobotStatus.IDLE and r.cell != (0, 0)


def test_charger_compatibility_and_power():
    d = tiny_dict(robot_types=types(agv={"charger_types": ["fast"]}),
                  charger_types=[{"name": "fast", "power_w": 1440.0}, {"name": "slow", "power_w": 360.0}])
    sim = Simulator(ScenarioConfig.from_dict(d))
    kinds = {c.id: c.type for c in sim.chargers}
    assert sorted(kinds.values()) == ["fast", "slow"]
    slow = [i for i, k in kinds.items() if k == "slow"][0]
    fast = [i for i, k in kinds.items() if k == "fast"][0]
    assert not sim.send_to_charger(0, slow)[0]
    assert sim.macro_action_mask(sim.robots[0])[1 + sim.K + slow] == 0
    assert sim.macro_action_mask(sim.robots[0])[1 + sim.K + fast] == 1
    assert sim.send_to_charger(2, slow)[0]                  # pickers accept both


def test_stranding_releases_and_penalizes_once(tiny):
    sim = tiny(robot_types=types(agv={"energy": {**AGV["energy"], "capacity_wh": 1.0}}),
               fleet=[{"type": "agv", "count": 1, "initial_soc": 0.1}, {"type": "picker", "count": 1}])
    park_all(sim)
    outs = run(sim, 4)
    r = sim.robots[0]
    assert r.status == RobotStatus.STRANDED and r.energy_wh == 0.0
    assert sum(o.robot_rewards[0] for o in outs) == pytest.approx(-1.0)
    assert not sim.robots[1].status == RobotStatus.STRANDED


# ───────────────────────────── speed and handling

def test_fractional_speed():
    d = tiny_dict(robot_types=types(agv={"speed": 0.5}))
    sim = Simulator(ScenarioConfig.from_dict(d))
    park_all(sim)
    place(sim, 0, (4, 0))
    sim.send_to_charger(0, 1)                               # far dock (0, 6)
    run(sim, 10)
    assert sim.robots[0].distance == 5


def test_handling_speed_scales_stage_time():
    d = tiny_dict(robot_types=types(agv={"handling_speed": 2.0}, picker={"handling_speed": 2.0}),
                  task_types=[retrieval(durations=(4, 1, 2))])
    sim = Simulator(ScenarioConfig.from_dict(d))
    park_all(sim)
    t = next(iter(sim.tasks.values()))
    place(sim, 0, t.origin)
    place(sim, 2, (t.origin[0], t.origin[1] - 1) if sim.layout.zone_of((t.origin[0], t.origin[1] - 1)) == "aisle"
          else (t.origin[0], t.origin[1] + 1))
    sim.assign(t.id, 0)
    sim.assign(t.id, 2)
    run(sim, 2)                                              # duration 4 / rate 2 = 2 ticks
    assert t.stage == 1 and t.carried


# ───────────────────────────── cooperation

def test_stage_waits_for_the_whole_crew(tiny):
    sim = tiny()
    park_all(sim)
    t = next(iter(sim.tasks.values()))
    sim.assign(t.id, 0)
    run(sim, 30)
    assert t.stage == 0 and t.progress == 0 and sim.robots[0].cell == t.origin
    assert not any(e.type == "stage_started" for e in sim.events.events)
    assert sim.assign(t.id, 2)[0]
    run(sim, 30)
    assert t.stage >= 1 and t.carried or t.done


def test_just_in_time_roles_and_full_completion(tiny):
    sim = tiny()
    park_all(sim)
    t = next(iter(sim.tasks.values()))
    sim.assign_coalition(t.id, {"carrier": [0], "picker": [2]})
    seen_deliver_with_picker = False
    for _ in range(200):
        sim.step([0] * len(sim.robots))
        if t.stage == 1:
            seen_deliver_with_picker |= bool(t.assigned["picker"])
            if not t.assigned["picker"]:
                assert "picker" in t.open_roles() and not t.joinable_roles()
        if t.stage == 2 and t.joinable_roles():
            sim.assign(t.id, 2)
        if t.done:
            break
    assert t.state == TaskState.COMPLETED
    assert not seen_deliver_with_picker                     # helper released while not needed
    types_ = [e.type for e in sim.events.events if e.task == t.id]
    for a, b in [("payload_picked", "payload_placed"), ("payload_placed", "task_completed")]:
        assert types_.index(a) < types_.index(b)
    for (tick, old, new, why) in t.record.history:
        assert new in ALLOWED_TRANSITIONS[old]
    assert all(r.status == RobotStatus.IDLE for r in sim.robots)


def test_joint_transport_moves_as_one_unit_and_shares_load():
    task = retrieval("job", pickers=0, weight=30.0)
    task["roles"][0]["count"] = 2
    sim = Simulator(ScenarioConfig.from_dict(tiny_dict(task_types=[task])))
    park_all(sim)
    t = next(iter(sim.tasks.values()))
    assert sim.assign(t.id, 0)[0]
    assert not t.carried
    assert sim.assign(t.id, 1)[0]
    carried_ticks = 0
    for _ in range(200):
        sim.step([0] * 4)
        if t.carried:
            carried_ticks += 1
            assert sim.robots[0].cell == sim.robots[1].cell
        if t.done:
            break
    assert t.done and carried_ticks > 0


def test_capacity_gates_who_can_carry():
    d = tiny_dict(robot_types=types() + [{**copy.deepcopy(AGV), "name": "mini", "payload_kg": 4.0}],
                  fleet=[{"type": "agv", "count": 1}, {"type": "mini", "count": 1}, {"type": "picker", "count": 2}])
    sim = Simulator(ScenarioConfig.from_dict(d))
    t = next(iter(sim.tasks.values()))                       # weight 5 kg
    assert not sim.can_join(sim.robots[1], t)[0]
    assert sim.can_join(sim.robots[0], t)[0]


def test_dropped_payload_is_recovered(tiny):
    sim = tiny()
    park_all(sim)
    t = next(iter(sim.tasks.values()))
    sim.assign_coalition(t.id, {"carrier": [0], "picker": [2]})
    for _ in range(100):
        sim.step([0] * 4)
        if t.carried and sim.robots[0].cell != t.origin:
            break
    assert t.carried
    r = sim.robots[0]
    r.status, r.repair_left, r.resume_status = RobotStatus.FAILED, 50, RobotStatus.TASK
    assert sim.macro_action_mask(r)[-1] == 1                # a failed robot may be released
    assert sim.release(0)[0]
    assert not t.carried and t.payload_cell == r.cell
    assert t.state == TaskState.SUSPENDED
    assert sim.assign(t.id, 1)[0]                            # a second AGV takes over at the drop point
    for _ in range(200):
        sim.step([0] * 4)
        if t.stage == 2 and t.joinable_roles():
            sim.assign(t.id, 3)
        if t.done:
            break
    assert t.state == TaskState.COMPLETED
    assert any(e.type == "payload_picked" and e.data.get("reload") for e in sim.events.events)


def test_random_breakdowns_and_repair():
    d = tiny_dict(robot_types=types(agv={"failure": {"rate": 0.2, "repair_ticks": 4}}))
    sim = Simulator(ScenarioConfig.from_dict(d))
    park_all(sim)
    run(sim, 60)
    ev = [e.type for e in sim.events.events]
    assert "robot_failed" in ev and "robot_repaired" in ev


# ───────────────────────────── demand and dependencies

def test_job_dependencies_and_location_chaining():
    scan = {"name": "scan", "roles": [{"name": "checker", "capability": "pick"}],
            "stages": [{"name": "scan", "at": "origin", "roles": ["checker"], "duration": 1}]}
    d = tiny_dict(task_types=[retrieval(), scan],
                  jobs=[{"name": "inbound", "tasks": [{"key": "a", "type": "job"},
                                                      {"key": "b", "type": "scan", "depends_on": ["a"], "origin_from": "a.origin"}]}],
                  demand=[{"kind": "batch", "job": "inbound", "count": 1}])
    sim = Simulator(ScenarioConfig.from_dict(d))
    a, b = sorted(sim.tasks.values(), key=lambda t: t.id)
    assert b.state == TaskState.PENDING and b.depends_on == (a.id,) and b.origin == a.origin
    assert b not in sim.open_tasks()
    a.record.transition(TaskState.ASSIGNED, 0); a.record.transition(TaskState.ACTIVE, 0)
    sim._complete(a, np.zeros(4))
    assert b.state == TaskState.READY


@pytest.mark.parametrize("kind", ["backlog", "trace", "poisson", "batch"])
def test_demand_processes(kind):
    stream = {"backlog": {"kind": "backlog", "task_type": "job", "open": 2},
              "trace": {"kind": "trace", "task_type": "job", "trace": [0, 5, 10]},
              "poisson": {"kind": "poisson", "task_type": "job", "rates": [[0, 0.0], [20, 2.0]], "max_items": 6},
              "batch": {"kind": "batch", "task_type": "job", "count": 2}}[kind]
    sim = Simulator(ScenarioConfig.from_dict(tiny_dict(demand=[stream])))
    if kind == "backlog":
        assert len(sim.open_tasks()) == 2
        t = sim.open_tasks()[0]
        t.record.transition(TaskState.CANCELLED, 0, reason="test")
        sim.step([0] * 4)
        assert len(sim.open_tasks()) == 2 and len(sim.tasks) == 3
    elif kind == "trace":
        run(sim, 12)
        assert sorted(t.released for t in sim.tasks.values()) == [0, 5, 10]
    elif kind == "poisson":
        run(sim, 19)
        assert not sim.tasks
        run(sim, 30)
        assert len(sim.tasks) == 6 and min(t.released for t in sim.tasks.values()) >= 20   # max_items
    else:
        assert len(sim.tasks) == 2 and not sim.demand_finished()
        for t in sim.tasks.values():
            t.record.transition(TaskState.CANCELLED, 0, reason="test")
        assert sim.demand_finished()


# ───────────────────────────── information

def test_sensing_communication_and_noise():
    import battery_tarware as bt
    d = tiny_dict(robot_types=types(picker={"sensing_range": 1, "comm_range": 0, "localization_noise": 0.0}),
                  fleet=[{"type": "agv", "count": 2}, {"type": "picker", "count": 2}])
    env = bt.make(d)
    env.reset(seed=0)
    sim = env.sim
    for i, c in enumerate([(1, 0), (5, 6), (1, 1), (4, 6)]):
        place(sim, i, c)
    vis = env.obs_builder.visibility()
    assert vis[2, 0] and not vis[2, 1]                      # picker sees its neighbour, not the far AGV
    assert vis[0, 1]                                        # AGVs: global sensing
    # communication relay: the picker hears what an in-range AGV sees
    d["robot_types"][1].update(comm_range=5)
    d["robot_types"][0].update(comm_range=5)
    env2 = bt.make(d)
    env2.reset(seed=0)
    for i, c in enumerate([(1, 0), (5, 6), (1, 1), (4, 6)]):
        place(env2.sim, i, c)
    assert env2.obs_builder.visibility()[2, 1]
    # localization noise only perturbs the noisy robot's observations
    d["robot_types"][1].update(localization_noise=1.0)
    env3 = bt.make(d)
    o, _ = env3.reset(seed=0)
    clean = env3.sim.robots[0].cell
    H, W = env3.sim.layout.shape
    assert o["agv_0"]["self"][0] == pytest.approx(clean[0] / (H - 1))
    noisy = [env3.obs_builder.build(env3.sim.action_masks())["picker_0"]["self"][0] for _ in range(5)]
    assert len(set(np.round(noisy, 6))) > 1


def test_task_visibility_by_sensing():
    d = tiny_dict(robot_types=types(picker={"sensing_range": 0}), observation={"task_visibility": "sensing"})
    sim = Simulator(ScenarioConfig.from_dict(d))
    park_all(sim)
    t = next(iter(sim.tasks.values()))
    far = 2 if sim.chebyshev(sim.robots[2].cell, t.stage_cell) > 0 else 3
    assert not sim.task_visible(sim.robots[far], t)
    assert sim.macro_action_mask(sim.robots[far])[1] == 0
    assert sim.task_visible(sim.robots[0], t)               # AGVs sense everything


# ───────────────────────────── primitive mode

def test_primitive_mode_moves_interacts_and_carries():
    from battery_tarware.simulator import P_DOWN, P_INTERACT, P_LEFT, P_RIGHT, P_UP, P_WAIT
    sim = Simulator(ScenarioConfig.from_dict(tiny_dict(action_mode="primitive",
                                                       fleet=[{"type": "agv", "count": 2, "initial_soc": 0.5},
                                                              {"type": "picker", "count": 2}],
                                                       task_types=[retrieval(pickers=0, durations=(1, 1, 1))])))
    park_all(sim)
    place(sim, 0, (1, 0))
    m = sim.primitive_action_mask(sim.robots[0])
    assert m[P_LEFT] == 0 and m[P_RIGHT] == 1
    sim.step([P_UP, P_WAIT, P_WAIT, P_WAIT])
    assert sim.robots[0].cell == (0, 0)
    sim.step([P_INTERACT, P_WAIT, P_WAIT, P_WAIT])          # on a dock: charge
    assert sim.robots[0].status == RobotStatus.CHARGE
    t = next(iter(sim.tasks.values()))
    place(sim, 1, t.origin)
    sim.step([P_WAIT, P_INTERACT, P_WAIT, P_WAIT])          # on the task cell: join
    assert sim.robots[1].task_id == t.id
    run(sim, 1, [P_WAIT] * 4)                               # pick stage (duration 1) done
    assert t.carried
    start = sim.robots[1].cell
    moves = [a for a in (P_UP, P_DOWN, P_LEFT, P_RIGHT) if sim.primitive_action_mask(sim.robots[1])[a]]
    sim.step([P_WAIT, moves[0], P_WAIT, P_WAIT])
    assert sim.robots[1].cell != start


# ───────────────────────────── determinism and views

def test_same_seed_same_trajectory_different_seed_differs():
    import battery_tarware as bt
    def trace(seed):
        env = bt.make("intermediate", max_steps=300)
        coord = bt.GreedyCoalitionCoordinator()
        env.reset(seed=seed)
        out = []
        for _ in range(300):
            o, r, te, tr, info = env.step(coord.act(env))
            out.append((round(r, 9), tuple((e.type, e.robots, e.task) for e in info["events"])))
        return out, env.sim.metrics()
    a, b, c = trace(7), trace(7), trace(8)
    assert a == b and a != c


def test_state_view_is_observation_only():
    import battery_tarware as bt
    def run_(with_views):
        env = bt.make("basic", max_steps=200)
        coord = bt.GreedyCoalitionCoordinator()
        env.reset(seed=3)
        for _ in range(200):
            if with_views:
                env.state_view().to_arrays()
            env.step(coord.act(env))
        return env.sim.metrics(), [(e.type, e.robots, e.task) for e in env.sim.events.events]
    assert run_(False) == run_(True)


def test_state_view_arrays_are_typed_and_contiguous():
    import battery_tarware as bt
    env = bt.make("advanced")
    env.reset(seed=0)
    a = env.state_view().to_arrays()
    assert a["schema_version"][0] == 1
    assert a["robot_cell"].dtype == np.int32 and a["robot_cell"].shape == (len(env.sim.robots), 2)
    assert a["robot_capabilities"].shape[1] == len(env.sim.capabilities)
    assert all(v.flags.c_contiguous for v in a.values())
