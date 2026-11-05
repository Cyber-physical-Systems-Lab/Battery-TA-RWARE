"""StateView: read-only, value-typed, and observation-only (does not perturb a trajectory)."""
import dataclasses
import hashlib

import numpy as np
import pytest

import battery_tarware as bt
from battery_tarware.interfaces import SCHEMA_VERSION, StateView, build_state_view


def _env():
    return bt.make_tiled_warehouse(tile_rows=1, tile_cols=2, num_agvs=4, num_pickers=2, max_steps=300)


def _rollout(view_every_step: bool, steps=150, seed=11):
    env = _env()
    env.reset(seed=seed)
    drv = np.random.default_rng([seed, 1])
    ids = sorted(int(k) for k in env.action_id_to_coords_map)
    h = hashlib.sha256()
    for _ in range(steps):
        if view_every_step:
            build_state_view(env)
        acts = [0 if drv.random() < 0.3 else ids[int(drv.integers(len(ids)))] for _ in env.agents]
        done, info = env.step(acts)
        h.update(repr([(a.x, a.y, a.battery, a.busy, a.path) for a in env.agents]).encode())
        h.update(repr(np.random.get_state()[1][:4]).encode())
        if done:
            break
    return h.hexdigest()


def test_view_does_not_perturb_trajectory_or_rng():
    assert _rollout(False) == _rollout(True)


def test_view_is_value_typed_and_frozen():
    env = _env()
    env.reset(seed=2)
    v = build_state_view(env)
    assert isinstance(v, StateView) and v.schema_version == SCHEMA_VERSION == 1
    assert len(v.robots) == len(env.agents) and len(v.shelves) == len(env.shelfs)
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.tick = 5
    # no live objects anywhere in the view
    def walk(o):
        if dataclasses.is_dataclass(o):
            for f in dataclasses.fields(o):
                yield from walk(getattr(o, f.name))
        elif isinstance(o, tuple):
            for x in o:
                yield from walk(x)
        else:
            yield o
    assert all(type(x) in (int, float, bool, bytes) for x in walk(v))
    before = repr(v)
    env.step([0] * len(env.agents))
    assert repr(v) == before                       # later steps never change an existing view


def test_view_matches_kernel_fields():
    env = _env()
    env.reset(seed=4)
    for _ in range(5):
        env.step([0] * len(env.agents))
    v = build_state_view(env)
    a0 = env.agents[0]
    r0 = v.robot(a0.id)
    assert (r0.x, r0.y, r0.battery, r0.kind) == (a0.x, a0.y, float(a0.battery), a0.type.value)
    assert v.request_queue == tuple(s.id for s in env.request_queue)
    assert v.tick == env._cur_steps
    for i in range(env.grid.shape[0]):
        assert (v.layer(i) == env.grid[i]).all()


def test_layout_view():
    from battery_tarware.interfaces import build_layout_view
    env = _env()
    env.reset(seed=1)
    lv = build_layout_view(env)
    assert lv.grid_shape == tuple(env.grid_size)
    assert len(lv.highways) == env.grid_size[0] * env.grid_size[1]
    assert dict((a, (y, x)) for a, y, x in lv.action_coords) == {int(k): tuple(map(int, v)) for k, v in env.action_id_to_coords_map.items()}
