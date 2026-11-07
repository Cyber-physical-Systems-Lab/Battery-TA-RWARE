"""Order-layer scenarios on the three built-in warehouses (not registered as benchmark tiers; the
built-in tiers and their demand streams are frozen).

``order_scenario(base=...)`` replaces a built-in scenario's demand with an ``orders:`` section.
Outbound and inbound orders bring a shelf to a station and back, relocations move a shelf to another
rack location. The task type of each operation is chosen by shelf weight, so one order stream
exercises several coalition compositions:

basic         AGV alone (solo, <= 6 kg), AGV + 1 picking robot (standard, <= 12 kg), AGV + 2 picking
              robots (large); relocation by one AGV.
intermediate  tote or pallet AGV alone (tote, <= 8 kg); carrier + 1 support, where the support is a
              picking robot on the shelf cell *or* a tote AGV with ``load_assist`` beside it (case,
              <= 20 kg; role alternatives); pallet AGV + 2 picking robots (pallet, <= 55 kg); two
              carriers + a spotter (oversize); relocation by one carrier (<= 60 kg) or two. Fast and
              slow docks with compatibility limits (pallet AGVs charge only on slow docks).
advanced      AGV + mobile manipulator + inventory drone with a quality-check stage in the QC zone
              (checked, <= 25 kg); two AGVs + manipulator (bulky, <= 60 kg); forklift + manipulator
              (pallet); inbound to the receiving zone by forklift + manipulator; relocation by an AGV
              (<= 30 kg) or a forklift. Breakdowns; drones with small batteries on fast docks only.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from battery_tarware.scenarios import get_scenario

RELOCATE = {
    "name": "relocate", "carrier_role": "carrier", "origin": "storage", "destination": "storage",
    "roles": [{"name": "carrier", "capability": "transport", "presence": "on_cell"}],
    "weight": {"kind": "fixed", "value": 0.0},
    "stages": [{"name": "lift", "at": "origin", "roles": ["carrier"], "duration": 2},
               {"name": "place", "at": "destination", "roles": ["carrier"], "duration": 2}],
}

BY_WEIGHT = [{"max_kg": 6.0, "task_type": "solo"}, {"max_kg": 12.0, "task_type": "standard"},
             {"task_type": "large"}]


RELOCATE_JOINT = {
    "name": "relocate_joint", "carrier_role": "carriers", "origin": "storage", "destination": "storage",
    "roles": [{"name": "carriers", "capability": "transport", "count": 2, "presence": "on_cell"}],
    "weight": {"kind": "fixed", "value": 0.0},
    "stages": [{"name": "lift", "at": "origin", "roles": ["carriers"], "duration": 3},
               {"name": "place", "at": "destination", "roles": ["carriers"], "duration": 3}],
}

_W0 = {"kind": "fixed", "value": 0.0}          # order-layer task types: the weight is the shelf's

ADVANCED_TYPES = [
    {"name": "checked", "carrier_role": "carrier", "weight": _W0,
     "roles": [{"name": "carrier", "capability": "transport", "presence": "on_cell"},
               {"name": "handler", "capability": "manipulate"},
               {"name": "checker", "capability": "inspect"}],
     "stages": [{"name": "pick", "at": "origin", "roles": ["carrier", "handler"], "duration": 3},
                {"name": "quality_check", "at": "zone:qc", "roles": ["carrier", "checker"], "duration": 3},
                {"name": "pack", "at": "destination", "roles": ["carrier", "handler"], "duration": 4},
                {"name": "return", "at": "origin", "roles": ["carrier"], "duration": 2}]},
    {"name": "bulky", "carrier_role": "carriers", "weight": _W0,
     "roles": [{"name": "carriers", "capability": "transport", "count": 2, "presence": "on_cell"},
               {"name": "handler", "capability": "manipulate"}],
     "stages": [{"name": "lift", "at": "origin", "roles": ["carriers", "handler"], "duration": 4},
                {"name": "pack", "at": "destination", "roles": ["carriers", "handler"], "duration": 5},
                {"name": "return", "at": "origin", "roles": ["carriers"], "duration": 3}]},
    {"name": "pallet", "carrier_role": "lifter", "weight": _W0,
     "roles": [{"name": "lifter", "capability": "lift", "presence": "on_cell"},
               {"name": "handler", "capability": "manipulate"}],
     "stages": [{"name": "lift", "at": "origin", "roles": ["lifter"], "duration": 3},
                {"name": "handle", "at": "destination", "roles": ["lifter", "handler"], "duration": 5},
                {"name": "return", "at": "origin", "roles": ["lifter"], "duration": 3}]},
    {"name": "relocate_agv", "carrier_role": "carrier", "origin": "storage", "destination": "storage",
     "weight": _W0, "roles": [{"name": "carrier", "capability": "transport", "presence": "on_cell"}],
     "stages": [{"name": "lift", "at": "origin", "roles": ["carrier"], "duration": 2},
                {"name": "place", "at": "destination", "roles": ["carrier"], "duration": 2}]},
]

DEFAULTS = {
    "basic": {
        "operations": {"outbound": {"task_types": BY_WEIGHT, "station": "goal"},
                       "inbound": {"task_types": BY_WEIGHT, "station": "goal"},
                       "relocation": {"task_types": [{"task_type": "relocate"}]}},
        "inventory": {"shelf_fill": 0.85, "package_fill": 0.6, "slot_capacity": 1,
                      "package_weight": {"kind": "uniform", "low": 0.5, "high": 16.0}},
    },
    "intermediate": {
        "operations": {
            "outbound": {"task_types": [{"max_kg": 8.0, "task_type": "tote"}, {"max_kg": 20.0, "task_type": "case"},
                                        {"max_kg": 55.0, "task_type": "pallet"}, {"task_type": "oversize"}],
                         "station": "goal"},
            "inbound": {"task_types": [{"max_kg": 8.0, "task_type": "tote"}, {"max_kg": 20.0, "task_type": "case"},
                                       {"max_kg": 55.0, "task_type": "pallet"}, {"task_type": "oversize"}],
                        "station": "goal"},
            "relocation": {"task_types": [{"max_kg": 60.0, "task_type": "relocate"},
                                          {"task_type": "relocate_joint"}]}},
        "inventory": {"shelf_fill": 0.8, "package_fill": 0.5, "slot_capacity": 2,
                      "package_weight": {"kind": "normal", "mean": 9.0, "std": 7.0, "low": 0.5, "high": 35.0}},
    },
    "advanced": {
        "operations": {
            "outbound": {"task_types": [{"max_kg": 25.0, "task_type": "checked"}, {"max_kg": 60.0, "task_type": "bulky"},
                                        {"task_type": "pallet"}], "station": "goal"},
            "inbound": {"task_types": [{"task_type": "pallet"}], "station": "receiving"},
            "relocation": {"task_types": [{"max_kg": 30.0, "task_type": "relocate_agv"},
                                          {"task_type": "relocate"}]}},
        "inventory": {"shelf_fill": 0.8, "package_fill": 0.5, "slot_capacity": 2,
                      "package_weight": {"kind": "normal", "mean": 10.0, "std": 9.0, "low": 0.5, "high": 120.0}},
    },
}


def order_scenario(source: Optional[Dict[str, Any]] = None, inventory: Optional[Dict[str, Any]] = None,
                   lifecycle: Optional[Dict[str, Any]] = None, base: str = "basic", model: str = "2.1",
                   substitution: Optional[bool] = None, **overrides) -> Dict[str, Any]:
    """An order scenario on the built-in warehouse ``base`` (basic | intermediate | advanced).

    ``model``: simulation model version (``"2.1"`` default, ``"2.2"`` opt-in). ``substitution``
    (intermediate only, default True): the ``case`` support role may be filled by a picking robot on
    the shelf cell or by a tote AGV with ``load_assist`` from the next cell."""
    if base not in DEFAULTS:
        raise ValueError("order_scenario: base must be one of %s" % sorted(DEFAULTS))
    if base == "intermediate":
        cfg = copy.deepcopy(get_scenario(base, model=model, substitution=True if substitution is None else substitution))
    else:
        cfg = copy.deepcopy(get_scenario(base, model=model))
    cfg["name"] = "%s_orders" % base
    cfg["description"] = "%s warehouse driven by inbound / outbound / relocation orders" % base
    if base == "basic":
        cfg["task_types"] = list(cfg["task_types"]) + [copy.deepcopy(RELOCATE)]
    elif base == "intermediate":
        cfg["task_types"] = list(cfg["task_types"]) + [copy.deepcopy(RELOCATE), copy.deepcopy(RELOCATE_JOINT)]
    else:
        keep = [t for t in cfg["task_types"] if t["name"] == "relocate"]
        cfg["task_types"] = keep + copy.deepcopy(ADVANCED_TYPES)
        cfg["jobs"] = []
        # the left rack column of the advanced map borders the receiving and QC zones only; vehicles
        # may drive (loaded) through both open floor areas, so every rack location is accessible
        for rt in cfg["robot_types"]:
            if rt["name"] in ("agv", "forklift"):
                for k in ("zones", "zones_loaded"):
                    rt[k] = list(dict.fromkeys(list(rt[k]) + ["receiving", "qc"]))
    cfg["demand"] = []
    dflt = copy.deepcopy(DEFAULTS[base])
    cfg["orders"] = {
        "operations": dflt["operations"],
        "inventory": {**dflt["inventory"], **(inventory or {})},
        "source": source or {"kind": "poisson", "rate": 0.02, "max_orders": 40,
                             "mix": {"outbound": 0.6, "inbound": 0.25, "relocation": 0.15},
                             "due_after": {"kind": "uniform", "low": 300, "high": 600}},
        "lifecycle": dict(lifecycle or {}),
    }
    cfg.update(overrides)
    return cfg
