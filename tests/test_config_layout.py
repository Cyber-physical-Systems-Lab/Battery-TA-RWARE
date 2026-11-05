import json
import random

import numpy as np
import pytest

import battery_tarware as bt
from battery_tarware import ConfigError, ScenarioConfig, get_scenario
from battery_tarware.layout import Layout, generate_ascii
from battery_tarware.navigation import astar
from conftest import retrieval, tiny_dict


@pytest.mark.parametrize("name", sorted(bt.SCENARIOS))
def test_builtin_scenarios_validate_and_round_trip(name, tmp_path):
    cfg = ScenarioConfig.from_dict(get_scenario(name))
    p = tmp_path / "s.json"
    p.write_text(json.dumps(cfg.to_dict(), default=list))
    again = ScenarioConfig.load(str(p))
    assert again.to_dict() == cfg.to_dict()


def test_yaml_scenario(tmp_path):
    yaml = pytest.importorskip("yaml")
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(tiny_dict()))
    assert bt.make(str(p)).config.name == "tiny"


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(bogus=1), "unknown key"),
    (lambda d: d["robot_types"][0].update(speed=1.5), "speed"),
    (lambda d: d["robot_types"][0].update(zones=["aisle", "lava"]), "unknown zone"),
    (lambda d: d["task_types"][0]["stages"][1].update(roles=["picker"]), "carrier role must take part"),
    (lambda d: d["task_types"][0].update(weight=50.0), "lift"),
    (lambda d: d["task_types"][0]["roles"][1].update(count=3), "needs 3 robot"),
    (lambda d: d["fleet"][0].update(initial_soc=1.5), "initial_soc"),
    (lambda d: d.update(jobs=[{"name": "j", "tasks": [{"key": "a", "type": "job", "depends_on": ["b"]},
                                                      {"key": "b", "type": "job", "depends_on": ["a"]}]}]), "cycle"),
    (lambda d: d["demand"][0].update(task_type="nope"), "unknown task type"),
    (lambda d: d["robot_types"][0].update(charger_types=["warp"]), "unknown charger"),
])
def test_validation_rejects_bad_configs(mutate, message):
    d = tiny_dict()
    mutate(d)
    with pytest.raises(ConfigError, match=message):
        ScenarioConfig.from_dict(d)


def test_group_overrides_make_robots_of_one_type_differ():
    d = tiny_dict(fleet=[{"type": "agv", "count": 1},
                         {"type": "agv", "name": "old_agv", "count": 1,
                          "overrides": {"speed": 0.5, "energy": {"capacity_wh": 400.0}}},
                         {"type": "picker", "count": 2}])
    specs = {n: s for n, s, _ in ScenarioConfig.from_dict(d).resolved_robot_specs()}
    assert specs["agv_0"].speed == 1.0 and specs["old_agv_0"].speed == 0.5
    assert specs["old_agv_0"].energy.capacity_wh == 400.0
    assert specs["old_agv_0"].energy.move_w == specs["agv_0"].energy.move_w   # untouched fields kept


def test_distributions():
    rng = np.random.default_rng(0)
    D = bt.Distribution
    n = D.from_any({"kind": "normal", "mean": 5, "std": 3, "low": 1, "high": 8}, "w")
    xs = [n.sample(rng) for _ in range(2000)]
    assert min(xs) >= 1 and max(xs) <= 8
    c = D.from_any({"kind": "choice", "values": [1, 2], "weights": [1, 3]}, "w")
    ys = [c.sample(rng) for _ in range(2000)]
    assert set(ys) == {1.0, 2.0} and 0.65 < ys.count(2.0) / 2000 < 0.85


@pytest.mark.parametrize("gen", [{}, {"rack_rows": 3, "rack_cols": 4}, {"rack_rows": 1, "rack_cols": 1, "rack_height": 2},
                                 {"rack_rows": 2, "rack_cols": 5, "rack_height": 3, "goals": 4, "docks": 6},
                                 {"aisle_width": 1, "one_way": True},
                                 {"aisle_width": 1, "one_way": True, "rack_rows": 3, "rack_cols": 4},
                                 {"aisle_width": 1, "one_way": True, "rack_rows": 1, "rack_cols": 1, "rack_height": 2},
                                 {"aisle_width": 3, "rack_cols": 2}])
def test_generated_maps_are_strongly_connected(gen):
    lay = Layout.from_ascii(generate_ascii(**gen), {}, ("aisle", "storage", "goal", "dock"))
    assert lay.one_way == bool(gen.get("one_way"))
    for zones in [("aisle", "storage", "goal", "dock"), ("aisle", "goal", "dock")]:
        pas = lay.passable_mask(zones)
        cells = [tuple(map(int, c)) for c in zip(*np.nonzero(pas))]
        hub = cells[len(cells) // 2]
        for c in cells:
            assert astar(pas, hub, c, None, lay.directions) is not None, c
            assert astar(pas, c, hub, None, lay.directions) is not None, c


@pytest.mark.parametrize("name", sorted(bt.SCENARIOS))
def test_builtin_maps_strongly_connected_for_every_robot_type(name):
    sim = bt.make(name).sim
    for rt in sim.config.robot_types:
        pas = sim.layout.passable_mask(rt.zones)
        cells = [tuple(map(int, c)) for c in zip(*np.nonzero(pas))]
        hub = cells[len(cells) // 2]
        for c in cells:
            assert astar(pas, hub, c, None, sim.layout.directions) is not None, (rt.name, c)


def test_one_way_rules():
    lay = Layout.from_ascii(".SG\n>>>\n...", {}, ("aisle", "storage", "goal", "dock"))
    UP, DOWN, LEFT, RIGHT = range(4)
    assert lay.move_ok((1, 0), (1, 1), RIGHT)       # along the street
    assert not lay.move_ok((1, 1), (1, 0), LEFT)    # against it
    assert lay.move_ok((1, 1), (0, 1), UP)          # turn off into storage
    assert lay.move_ok((1, 1), (2, 1), DOWN)        # turn off into a plain (two-way) aisle cell
    assert lay.move_ok((0, 0), (1, 0), DOWN)        # merge onto the street from the side
    assert lay.move_ok((2, 2), (1, 2), UP)


def test_generator_rejects_one_way_with_wide_aisles():
    with pytest.raises(ConfigError):
        generate_ascii(aisle_width=2, one_way=True)


def test_custom_zone_symbols():
    lay = Layout.from_ascii("II.\n.SG", {"I": "receiving"}, ("aisle", "storage", "goal", "dock", "receiving"))
    assert lay.cells("receiving") == [(0, 0), (0, 1)]
    with pytest.raises(ConfigError):
        Layout.from_ascii("X..\n..G", {}, ("aisle", "storage", "goal", "dock"))
