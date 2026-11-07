"""Built-in scenarios (benchmark tiers) and scenario resolution.

Tiers add *explicit cooperation constraints*, not just more robots. The robot types follow roles
found in automated warehouses:

* ``agv`` / ``tote_agv`` / ``pallet_agv``: mobile robots that carry totes, shelves or pallets;
* ``picking_robot`` / ``fast_picking_robot``: mobile piece-picking robots, i.e. a mobile base
  with an arm and gripper. One drives up beside a shelf, picks the order items into the AGV's tote
  and puts returned items back. It cannot carry loads itself;
* ``forklift``: an autonomous forklift for heavy pallets (receiving, put-away, relocation);
* ``mobile_manipulator``: a mobile arm that handles goods at receiving, packing and quality check;
* ``inventory_drone``: an indoor drone that scans shelves (cycle counts) and checks orders.

basic         two robot roles (AGV, picking robot), retrieval tasks needing 1-3 robots, one charger
              type.
intermediate  heterogeneity within roles (tote vs pallet AGVs, fast vs standard picking robots),
              joint transport of heavy loads, peak/trough demand, fast and slow chargers with
              compatibility limits, limited sensing and communication for picking robots.
advanced      four robot types (AGV, forklift, mobile manipulator, inventory drone), custom zones
              (receiving, quality check), multi-stage dependency-linked jobs (inbound -> inspection),
              relocations, deadlines, breakdowns and noisy localization.

Every scenario is a plain dict; ``get_scenario(name)`` returns a copy you can edit, and
``ScenarioConfig.from_dict`` validates it.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Union

from battery_tarware.config import ConfigError, ScenarioConfig

AGV_ENERGY = {"capacity_wh": 1843.0, "idle_w": 20.0, "move_w": 142.0, "payload_w_per_kg": 1.7,
              "handling_w": 60.0, "max_charge_w": 1500.0}
PICKER_ENERGY = {"capacity_wh": 1000.0, "idle_w": 15.0, "move_w": 90.0, "payload_w_per_kg": 0.0,
                 "handling_w": 80.0, "max_charge_w": 1000.0}


def _retrieval(name: str, pickers: int, weight: Dict[str, float]) -> Dict[str, Any]:
    roles = [{"name": "carrier", "capability": "transport", "presence": "on_cell"}]
    crew = ["carrier"]
    if pickers:
        roles.append({"name": "picker", "capability": "pick", "count": pickers, "presence": "adjacent"})
        crew = ["carrier", "picker"]
    return {
        "name": name, "roles": roles, "carrier_role": "carrier", "weight": weight,
        "stages": [
            {"name": "pick", "at": "origin", "roles": crew, "duration": 3},
            {"name": "deliver", "at": "destination", "roles": ["carrier"], "duration": 2},
            {"name": "return", "at": "origin", "roles": crew, "duration": 3},
        ],
    }


BASIC: Dict[str, Any] = {
    "name": "basic",
    "description": "AGV + picking-robot roles; SOLO / STANDARD / LARGE retrieval; one charger type.",
    "map": {"generator": {"rack_rows": 2, "rack_cols": 3, "rack_height": 4, "goals": 2, "docks": 4}},
    "robot_types": [
        {"name": "agv", "capabilities": ["transport"], "speed": 1.0, "payload_kg": 25.0,
         "zones": ["aisle", "storage", "goal", "dock"], "zones_loaded": ["aisle", "goal", "dock"],
         "energy": AGV_ENERGY},
        {"name": "picking_robot", "capabilities": ["pick"], "speed": 1.0, "payload_kg": 0.0,
         "zones": ["aisle", "goal", "dock"], "energy": PICKER_ENERGY},
    ],
    "fleet": [{"type": "agv", "count": 6, "initial_soc": [0.5, 1.0]},
              {"type": "picking_robot", "count": 3, "initial_soc": [0.5, 1.0]}],
    "charger_types": [{"name": "standard", "power_w": 1000.0}],
    "task_types": [
        _retrieval("solo", 0, {"kind": "normal", "mean": 2.5, "std": 1.0, "low": 0.5, "high": 6.0}),
        _retrieval("standard", 1, {"kind": "normal", "mean": 5.0, "std": 2.0, "low": 0.5, "high": 12.0}),
        _retrieval("large", 2, {"kind": "normal", "mean": 8.75, "std": 2.0, "low": 2.0, "high": 20.0}),
    ],
    "demand": [{"kind": "backlog", "open": 8, "mix": {"solo": 0.3, "standard": 0.4, "large": 0.3}}],
    "max_steps": 2000,
}

INTERMEDIATE: Dict[str, Any] = {
    "name": "intermediate",
    "description": "Within-role heterogeneity, joint transport, peak demand, fast/slow chargers, limited sensing.",
    "map": {"generator": {"rack_rows": 3, "rack_cols": 4, "rack_height": 4, "goals": 3, "docks": 6}},
    "robot_types": [
        {"name": "tote_agv", "capabilities": ["transport"], "speed": 1.0, "payload_kg": 20.0,
         "zones": ["aisle", "storage", "goal", "dock"], "zones_loaded": ["aisle", "goal", "dock"],
         "energy": {**AGV_ENERGY, "capacity_wh": 1200.0, "move_w": 120.0}},
        {"name": "pallet_agv", "capabilities": ["transport"], "speed": 0.5, "payload_kg": 60.0,
         "zones": ["aisle", "storage", "goal", "dock"], "zones_loaded": ["aisle", "goal", "dock"],
         "charger_types": ["slow"], "energy": {**AGV_ENERGY, "capacity_wh": 2500.0, "move_w": 190.0,
                                                "payload_w_per_kg": 1.2, "max_charge_w": 1000.0}},
        {"name": "fast_picking_robot", "capabilities": ["pick"], "speed": 1.0, "handling_speed": 2.0,
         "zones": ["aisle", "goal", "dock"], "sensing_range": 6, "comm_range": 10,
         "energy": {**PICKER_ENERGY, "capacity_wh": 800.0, "handling_w": 120.0}},
        {"name": "picking_robot", "capabilities": ["pick"], "speed": 0.8, "handling_speed": 1.0,
         "zones": ["aisle", "goal", "dock"], "sensing_range": 6, "comm_range": 10, "energy": PICKER_ENERGY},
    ],
    "fleet": [
        {"type": "tote_agv", "count": 5, "initial_soc": [0.3, 1.0]},
        {"type": "tote_agv", "name": "tote_agv_old", "count": 1, "initial_soc": 0.6,
         "overrides": {"energy": {"capacity_wh": 800.0}}},                  # an aged battery
        {"type": "pallet_agv", "count": 2, "initial_soc": [0.5, 1.0]},
        {"type": "fast_picking_robot", "count": 2, "initial_soc": [0.4, 1.0]},
        {"type": "picking_robot", "count": 3, "initial_soc": [0.4, 1.0]},
    ],
    "charger_types": [{"name": "fast", "power_w": 3000.0, "share": 1.0},
                      {"name": "slow", "power_w": 800.0, "share": 2.0}],
    "task_types": [
        _retrieval("tote", 0, {"kind": "uniform", "low": 1.0, "high": 8.0}),
        _retrieval("case", 1, {"kind": "normal", "mean": 12.0, "std": 4.0, "low": 3.0, "high": 20.0}),
        {**_retrieval("pallet", 2, {"kind": "uniform", "low": 25.0, "high": 55.0}), "priority": 2.0},
        {   # two carriers lift together; a heavy AGV alone cannot take it either (> 60 kg possible)
            "name": "oversize", "priority": 3.0, "carrier_role": "carriers",
            "weight": {"kind": "uniform", "low": 45.0, "high": 75.0},
            "roles": [{"name": "carriers", "capability": "transport", "count": 2, "presence": "on_cell"},
                      {"name": "spotter", "capability": "pick", "count": 1}],
            "stages": [{"name": "lift", "at": "origin", "roles": ["carriers", "spotter"], "duration": 4},
                       {"name": "deliver", "at": "destination", "roles": ["carriers"], "duration": 3},
                       {"name": "return", "at": "origin", "roles": ["carriers", "spotter"], "duration": 4}],
        },
    ],
    "demand": [
        {"kind": "poisson", "mix": {"tote": 0.45, "case": 0.35, "pallet": 0.15, "oversize": 0.05},
         "rates": [[0, 0.02], [1000, 0.05], [2000, 0.02], [3000, 0.05]]},
    ],
    "observation": {"task_slots": 16},
    "max_steps": 4000,
}

ADVANCED_MAP = """
.C..C..C..C..C..C.
..................
..................
IISS..SS..SS..SS..
IISS..SS..SS..SS..
IISS..SS..SS..SS..
IISS..SS..SS..SS..
..................
..................
QQSS..SS..SS..SS..
QQSS..SS..SS..SS..
QQSS..SS..SS..SS..
QQSS..SS..SS..SS..
..................
..................
..G....G...G...G..
"""

ADVANCED: Dict[str, Any] = {
    "name": "advanced",
    "description": "AGV, forklift, mobile manipulator, inventory drone; receiving and QC zones; "
                   "dependency-linked inbound jobs; joint transport; deadlines; breakdowns; noisy localization.",
    "map": {"ascii": ADVANCED_MAP, "zones": {"I": "receiving", "Q": "qc"}},
    "robot_types": [
        {"name": "agv", "capabilities": ["transport"], "speed": 1.0, "payload_kg": 30.0,
         "zones": ["aisle", "storage", "goal", "dock", "qc"], "zones_loaded": ["aisle", "goal", "dock", "qc"],
         "energy": AGV_ENERGY, "failure": {"rate": 0.0002, "repair_ticks": 60}},
        {"name": "forklift", "capabilities": ["lift"], "speed": 0.4, "payload_kg": 300.0,
         "zones": ["aisle", "receiving", "storage", "goal", "dock"], "zones_loaded": ["aisle", "receiving", "goal", "dock"],
         "charger_types": ["slow"],
         "energy": {"capacity_wh": 6000.0, "idle_w": 40.0, "move_w": 900.0, "payload_w_per_kg": 2.0,
                    "handling_w": 600.0, "max_charge_w": 1500.0},
         "failure": {"rate": 0.0001, "repair_ticks": 120}},
        {"name": "mobile_manipulator", "capabilities": ["manipulate", "pick"], "speed": 0.6, "handling_speed": 1.5,
         "zones": ["aisle", "goal", "receiving", "qc", "dock"],
         "energy": {"capacity_wh": 1500.0, "idle_w": 30.0, "move_w": 160.0, "payload_w_per_kg": 0.0,
                    "handling_w": 250.0, "max_charge_w": 1500.0}},
        {"name": "inventory_drone", "capabilities": ["inspect"], "speed": 1.0,
         "zones": ["aisle", "storage", "qc", "dock", "receiving", "goal"],
         "sensing_range": 8, "comm_range": 8, "localization_noise": 0.5, "charger_types": ["fast"],
         "energy": {"capacity_wh": 300.0, "idle_w": 60.0, "move_w": 250.0, "payload_w_per_kg": 0.0,
                    "handling_w": 50.0, "max_charge_w": 3000.0}},
    ],
    "fleet": [
        {"type": "agv", "count": 6, "initial_soc": [0.4, 1.0]},
        {"type": "forklift", "count": 2, "initial_soc": [0.6, 1.0]},
        {"type": "mobile_manipulator", "count": 3, "initial_soc": [0.5, 1.0]},
        {"type": "inventory_drone", "count": 2, "initial_soc": [0.3, 1.0]},
    ],
    "charger_types": [{"name": "fast", "power_w": 3000.0, "share": 1.0},
                      {"name": "slow", "power_w": 1200.0, "share": 1.0}],
    "task_types": [
        {"name": "order", "carrier_role": "carrier", "deadline": 600, "priority": 1.0,
         "weight": {"kind": "normal", "mean": 8.0, "std": 4.0, "low": 0.5, "high": 25.0},
         "roles": [{"name": "carrier", "capability": "transport", "presence": "on_cell"},
                   {"name": "handler", "capability": "manipulate"},
                   {"name": "checker", "capability": "inspect"}],
         "stages": [{"name": "pick", "at": "origin", "roles": ["carrier", "handler"], "duration": 3},
                    {"name": "quality_check", "at": "zone:qc", "roles": ["carrier", "checker"], "duration": 3},
                    {"name": "pack", "at": "destination", "roles": ["carrier", "handler"], "duration": 4}]},
        {"name": "bulky_order", "carrier_role": "carriers", "deadline": 900, "priority": 2.0,
         "weight": {"kind": "uniform", "low": 35.0, "high": 55.0},
         "roles": [{"name": "carriers", "capability": "transport", "count": 2, "presence": "on_cell"},
                   {"name": "handler", "capability": "manipulate"}],
         "stages": [{"name": "pick", "at": "origin", "roles": ["carriers", "handler"], "duration": 4},
                    {"name": "pack", "at": "destination", "roles": ["carriers", "handler"], "duration": 5}]},
        {"name": "receive", "carrier_role": "lifter", "origin": "receiving", "destination": "storage",
         "weight": {"kind": "normal", "mean": 120.0, "std": 40.0, "low": 40.0, "high": 250.0},
         "roles": [{"name": "lifter", "capability": "lift", "presence": "on_cell"},
                   {"name": "unloader", "capability": "manipulate"}],
         "stages": [{"name": "unload", "at": "origin", "roles": ["lifter", "unloader"], "duration": 4},
                    {"name": "putaway", "at": "destination", "roles": ["lifter"], "duration": 3}]},
        {"name": "cycle_count", "priority": 0.5,
         "roles": [{"name": "inspector", "capability": "inspect", "presence": "on_cell"}],
         "stages": [{"name": "scan", "at": "origin", "roles": ["inspector"], "duration": 5}]},
        {"name": "relocate", "carrier_role": "lifter", "origin": "storage", "destination": "storage", "priority": 0.5,
         "weight": {"kind": "uniform", "low": 60.0, "high": 200.0},
         "roles": [{"name": "lifter", "capability": "lift", "presence": "on_cell"}],
         "stages": [{"name": "lift", "at": "origin", "roles": ["lifter"], "duration": 2},
                    {"name": "place", "at": "destination", "roles": ["lifter"], "duration": 2}]},
    ],
    "jobs": [
        {"name": "inbound", "tasks": [
            {"key": "receive", "type": "receive"},
            {"key": "inspect", "type": "cycle_count", "depends_on": ["receive"], "origin_from": "receive.destination"}]},
    ],
    "demand": [
        {"kind": "poisson", "mix": {"order": 0.8, "bulky_order": 0.2}, "rates": [[0, 0.03], [1500, 0.06], [3000, 0.03]]},
        {"kind": "poisson", "job": "inbound", "rate": 0.008},
        {"kind": "backlog", "task_type": "relocate", "open": 1},
    ],
    "rewards": {"task_completed": 1.0, "late_per_tick": -0.002, "stranded": -5.0},
    "observation": {"task_slots": 20},
    "max_steps": 5000,
}

SCENARIOS: Dict[str, Dict[str, Any]] = {"basic": BASIC, "intermediate": INTERMEDIATE, "advanced": ADVANCED}


# Model 2.1 (TA-RWARE semantics): a picking robot steps onto the shelf cell beside the AGV to load or
# unload it, instead of working from the next cell. These (task type, role) pairs become on_cell.
ON_CELL_21 = {"basic": {("standard", "picker"), ("large", "picker")},
              "intermediate": {("case", "picker"), ("pallet", "picker")}}


def get_scenario(name: str, model: str = "2.1", substitution: bool = False) -> Dict[str, Any]:
    """A deep copy of a built-in scenario dict (edit it, then pass it to make / from_dict).
    ``model="2.0"`` returns the scenario exactly as in model 2.0 (pickers work from the next cell).

    ``substitution=True`` (intermediate, model 2.1 only): the support role of a ``case`` task may be
    filled either by a picking robot on the shelf cell (as before) or by a tote AGV with the
    ``load_assist`` capability working from the next cell (radius 1). Only ``tote_agv`` gets
    ``load_assist``; task demand, crew size, stage durations and every other task type are unchanged.
    ``load_assist`` is an abstract benchmark capability (a support operation done from beside the
    shelf), not a validated physical loading behaviour."""
    if name not in SCENARIOS:
        raise ConfigError("unknown scenario %r; built-in: %s" % (name, sorted(SCENARIOS)))
    if substitution and (name != "intermediate" or model == "2.0"):
        raise ConfigError("substitution=True is defined for the intermediate scenario under model 2.1 or later")
    d = copy.deepcopy(SCENARIOS[name])
    if model == "2.0":
        d["model"] = "2.0"
        return d
    if model != "2.1":
        d["model"] = model                 # 2.2: same scenario as 2.1, plus the 2.2 movement rule
    for tt in d["task_types"]:
        roles = []
        for role in tt["roles"]:
            if (tt["name"], role["name"]) not in ON_CELL_21.get(name, ()):
                roles.append(role)
                continue
            # one picker steps onto the shelf cell (as in TA-RWARE, one picker per shelf location);
            # a second one, if the task needs two, works from the next cell
            n = int(role.get("count", 1))
            roles.append(dict(role, count=1, presence="on_cell"))
            roles[-1].pop("radius", None)
            if n > 1:
                extra = dict(role, name=role["name"] + "_2", count=n - 1, presence="adjacent")
                roles.append(extra)
                for st in tt["stages"]:
                    if role["name"] in st["roles"]:
                        st["roles"] = list(st["roles"]) + [extra["name"]]
        tt["roles"] = roles
    if substitution:
        d["name"] = "intermediate (substitutable support)"
        for rt in d["robot_types"]:
            if rt["name"] == "tote_agv":
                rt["capabilities"] = list(rt["capabilities"]) + ["load_assist"]
        for tt in d["task_types"]:
            if tt["name"] != "case":
                continue
            for i, role in enumerate(tt["roles"]):
                if role["name"] == "picker":
                    tt["roles"][i] = {"name": "picker", "count": role.get("count", 1), "alternatives": [
                        {"capability": "pick", "presence": "on_cell"},
                        {"capability": "load_assist", "presence": "adjacent", "radius": 1}]}
    return d


def resolve_scenario(scenario: Union[str, dict, ScenarioConfig] = "basic", **overrides: Any) -> ScenarioConfig:
    """Name of a built-in scenario, path to a .json/.yaml file, dict, or ScenarioConfig; top-level
    fields can be overridden by keyword (e.g. ``max_steps=500, action_mode="primitive"``)."""
    if isinstance(scenario, ScenarioConfig):
        cfg = scenario
        return cfg.replace(**overrides) if overrides else cfg
    if isinstance(scenario, str):
        if scenario in SCENARIOS:
            d = get_scenario(scenario, model=overrides.get("model", "2.1"))
        elif Path(scenario).suffix in (".json", ".yaml", ".yml"):
            cfg = ScenarioConfig.load(scenario)
            return cfg.replace(**overrides) if overrides else cfg
        else:
            raise ConfigError("unknown scenario %r (built-in: %s, or a .json/.yaml path)" % (scenario, sorted(SCENARIOS)))
    else:
        d = copy.deepcopy(dict(scenario))
    d.update(overrides)
    return ScenarioConfig.from_dict(d)
