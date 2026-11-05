"""Gymnasium / PettingZoo conformance, registration, rendering, coordinators, packaging."""
import ast
import os
import subprocess
import sys
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import battery_tarware as bt
from battery_tarware.coordination import RULES

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", sorted(bt.SCENARIOS))
@pytest.mark.parametrize("mode", ["macro", "primitive"])
def test_gymnasium_check_env(name, mode):
    env = gym.make(bt.ENV_IDS[name], action_mode=mode, max_steps=200)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # any checker warning fails the test
        check_env(env.unwrapped)


@pytest.mark.parametrize("env_id", list(bt.ENV_IDS.values()))
def test_registered_ids(env_id):
    env = gym.make(env_id, max_steps=50)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if term or trunc:
            break
    assert trunc or term


def test_observations_stay_in_space_and_masks_match():
    env = bt.make("advanced", max_steps=300)
    obs, info = env.reset(seed=1)
    coord = bt.RandomCoordinator(seed=0, noop_prob=0.3)
    for _ in range(300):
        obs, r, term, trunc, info = env.step(coord.act(env))
        assert env.observation_space.contains(obs)
        for i, name in enumerate(env.robot_names):
            assert np.array_equal(obs[name]["action_mask"], info["action_mask"][i])
        assert np.isclose(r, info["robot_rewards"].sum())
        if term or trunc:
            break


def test_masked_actions_are_valid_and_conflicts_are_counted():
    """Masks are computed before the tick and robots act simultaneously: a masked action of one robot
    is always accepted; two robots grabbing the last slot of a task is a conflict, counted in
    info["invalid_actions"] (the later one, in fleet order, is rejected)."""
    env = bt.make("intermediate", max_steps=300)
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for t in range(300):
        masks = env.action_masks()
        act = np.zeros(len(masks), dtype=np.int64)
        i = t % len(masks)                                   # one robot acts per tick
        valid = np.flatnonzero(masks[i][1:]) + 1
        if len(valid):
            act[i] = rng.choice(valid)
        _, _, term, trunc, info = env.step(act)
        assert info["invalid_actions"] == 0
        if term or trunc:
            break
    env = bt.make("basic")                                    # starts with a backlog of open tasks
    env.reset(seed=0)
    masks = env.action_masks()
    slot = 1 + next(k for k in range(env.sim.K) if masks[:, 1 + k].sum() >= 2)
    t = env.sim.task_slots()[slot - 1]
    takers = [i for i in range(len(masks)) if masks[i, slot]][:2]
    roles = {env.sim.can_join(env.sim.robots[i], t)[1] for i in takers}
    act = np.zeros(len(masks), dtype=np.int64)
    act[takers] = slot
    _, _, _, _, info = env.step(act)
    if len(roles) == 1 and t.type.role(roles.pop()).count == 1:
        assert info["invalid_actions"] == 1


def test_pettingzoo_parallel_api():
    pz = pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test
    from battery_tarware.pettingzoo_env import BatteryTarwareParallelEnv
    for mode in ("macro", "primitive"):
        parallel_api_test(BatteryTarwareParallelEnv("basic", action_mode=mode, max_steps=120), num_cycles=120)
    env = BatteryTarwareParallelEnv("basic", reward_mode="team", max_steps=10)
    obs, infos = env.reset(seed=0)
    acts = {a: 0 for a in env.agents}
    obs, rew, term, trunc, infos = env.step(acts)
    assert len(set(rew.values())) == 1                       # team reward is shared
    assert env.state().ndim == 1


@pytest.mark.parametrize("mode", ["rgb_array", "ansi"])
def test_render(mode):
    env = bt.make("basic", render_mode=mode)
    env.reset(seed=0)
    out = env.render()
    if mode == "rgb_array":
        H, W = env.sim.layout.shape
        assert out.shape == (H * 16, W * 16, 3) and out.dtype == np.uint8
    else:
        assert out.startswith("tick 0")


def test_rules_registry_is_documented_and_orders_tasks():
    for r in RULES.values():
        assert r.full_name and len(r.definition) > 20
    env = bt.make("advanced")
    env.reset(seed=0)
    sim = env.sim
    for _ in range(50):
        sim.step([0] * len(sim.robots))
    ts = sim.open_tasks()
    fifo = sorted(ts, key=lambda t: RULES["fifo"].key(sim, t))
    assert [t.released for t in fifo] == sorted(t.released for t in ts)
    edd = sorted(ts, key=lambda t: RULES["edd"].key(sim, t))
    dl = [t.deadline for t in edd if t.deadline is not None]
    assert dl == sorted(dl)


@pytest.mark.parametrize("rule", sorted(RULES))
def test_greedy_coordinator_every_rule(rule):
    m = bt.run_episode(bt.make("basic", max_steps=600), bt.GreedyCoalitionCoordinator(rule), seed=2)
    assert m["tasks_completed"] > 10 and m["stranded"] == 0


def test_feasible_coalitions_respect_capability_capacity_and_energy():
    env = bt.make("intermediate")
    env.reset(seed=0)
    sim = env.sim
    for _ in range(300):
        sim.step([0] * len(sim.robots))
        heavy = [t for t in sim.open_tasks() if t.type.name in ("pallet", "oversize")]
        if heavy:
            break
    assert heavy
    for t in heavy:
        for c in sim.feasible_coalitions(t.id, limit=20):
            members = [sim.robots[i] for i in c.robots]
            assert len(set(c.robots)) == len(c.robots)
            for role, ids in c.assignment.items():
                cap = t.type.role(role).capability
                assert all(sim.robots[i].can(cap) for i in ids)
            carriers = c.assignment.get(t.type.carrier_role, ())
            assert sum(sim.robots[i].spec.payload_kg for i in carriers) >= t.weight_kg
            for r in members:
                assert r.energy_wh - c.est_energy_wh[r.id] >= 0.05 * r.spec.energy.capacity_wh


FORBIDDEN = {"tarware", "planner", "sh", "analysis", "torch"}


def test_package_imports_nothing_from_research_code():
    bad = []
    for py in (ROOT / "battery_tarware").rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else []
            bad += ["%s: %s" % (py.name, n) for n in names if n.split(".")[0] in FORBIDDEN]
    assert not bad, bad


def test_base_install_does_not_import_optional_dependencies():
    code = ("import sys, battery_tarware as bt; env = bt.make('basic'); env.reset(seed=0); env.step([0]*len(env.robot_names));"
            "bad = [m for m in ('pettingzoo', 'matplotlib', 'yaml', 'torch') if m in sys.modules]; assert not bad, bad")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
