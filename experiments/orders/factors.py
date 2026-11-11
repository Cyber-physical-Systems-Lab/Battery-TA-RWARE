"""Scenario factors for the order-scheduling evaluation (basic warehouse with an orders: section).

``build(**levels)`` returns a scenario dict; ``BASE`` gives the base level of every factor and
``FACTORS`` the alternative levels varied one at a time (exploratory design). Every level is a
plain configuration change of existing simulator features; nothing is added to favour a rule.

factor        levels (base first)
rate          0.09 orders/tick | 0.05 | 0.13
mix           balanced (out 0.45 / in 0.45 / reloc 0.10) | outbound_heavy (0.6/0.3/0.1) | relocation_heavy (0.35/0.35/0.3)
weight        mixed uniform 0.5-16 kg (solo/standard/large crews) | light 0.5-6 kg (AGV alone) | heavy 6-20 kg (AGV + 1-2 pickers)
fleet         homogeneous (6 AGV, 3 picking robots) | mixed (4 AGV, 2 slow heavy AGVs, 2 picking, 1 fast picking robot)
battery       default (OMRON-class packs, little charging) | tight (AGV 120 Wh, picking robot 80 Wh; 2 docks)
layout        basic (2x3 rack blocks, 2 goals) | wide (3x4 rack blocks, 3 goals)
"""
from __future__ import annotations

import copy

from battery_tarware.orders.scenarios import order_scenario

BASE = {"rate": 0.09, "mix": "balanced", "weight": "mixed", "fleet": "homogeneous", "battery": "default",
        "layout": "basic"}
FACTORS = {"rate": [0.05, 0.13], "mix": ["outbound_heavy", "relocation_heavy"], "weight": ["light", "heavy"],
           "fleet": ["mixed"], "battery": ["tight"], "layout": ["wide"]}
MIX = {"balanced": {"outbound": 0.45, "inbound": 0.45, "relocation": 0.10},
       "outbound_heavy": {"outbound": 0.6, "inbound": 0.3, "relocation": 0.1},
       "relocation_heavy": {"outbound": 0.35, "inbound": 0.35, "relocation": 0.3}}
WEIGHT = {"mixed": {"kind": "uniform", "low": 0.5, "high": 16.0}, "light": {"kind": "uniform", "low": 0.5, "high": 6.0},
          "heavy": {"kind": "uniform", "low": 6.0, "high": 20.0}}


def build(rate=BASE["rate"], mix=BASE["mix"], weight=BASE["weight"], fleet=BASE["fleet"], battery=BASE["battery"],
          layout=BASE["layout"], max_steps=3000):
    src = {"kind": "poisson", "rate": float(rate), "max_orders": 10 ** 6, "mix": MIX[mix],
           "priority": {"kind": "choice", "values": [1, 2, 3]}, "due_after": {"kind": "uniform", "low": 300, "high": 800}}
    cfg = order_scenario(source=src, inventory={"shelf_fill": 0.85, "package_fill": 0.5, "slot_capacity": 2,
                                                "package_weight": WEIGHT[weight]})
    cfg["max_steps"] = max_steps
    if layout == "wide":
        cfg["map"] = {"generator": {"rack_rows": 3, "rack_cols": 4, "rack_height": 4, "goals": 3, "docks": 4}}
    if fleet == "mixed":
        agv = copy.deepcopy(cfg["robot_types"][0])
        heavy = {**copy.deepcopy(agv), "name": "heavy_agv", "speed": 0.5, "payload_kg": 60.0,
                 "energy": {**agv["energy"], "capacity_wh": 2500.0, "move_w": 190.0}}
        pick = copy.deepcopy(cfg["robot_types"][1])
        fast = {**copy.deepcopy(pick), "name": "fast_picking_robot", "handling_speed": 2.0,
                "energy": {**pick["energy"], "capacity_wh": 800.0, "handling_w": 120.0}}
        cfg["robot_types"] = [agv, heavy, pick, fast]
        cfg["fleet"] = [{"type": "agv", "count": 4, "initial_soc": [0.5, 1.0]},
                        {"type": "heavy_agv", "count": 2, "initial_soc": [0.5, 1.0]},
                        {"type": "picking_robot", "count": 2, "initial_soc": [0.5, 1.0]},
                        {"type": "fast_picking_robot", "count": 1, "initial_soc": [0.5, 1.0]}]
    if battery == "tight":
        for rt in cfg["robot_types"]:
            rt["energy"] = {**rt["energy"], "capacity_wh": 120.0 if rt["payload_kg"] > 0 else 80.0}
        gen = dict(cfg["map"]["generator"])
        gen["docks"] = 2
        cfg["map"] = {"generator": gen}
    cfg["name"] = "orders_" + "_".join("%s-%s" % (k, v) for k, v in
                                       dict(rate=rate, mix=mix, weight=weight, fleet=fleet, battery=battery,
                                            layout=layout).items() if v != BASE[k]) if any(
        v != BASE[k] for k, v in dict(rate=rate, mix=mix, weight=weight, fleet=fleet, battery=battery,
                                      layout=layout).items()) else "orders_base"
    return cfg


def configs(max_steps=3000):
    """Exploratory one-factor-at-a-time design: the base plus each alternative level."""
    out = {"base": build(max_steps=max_steps)}
    for f, levels in FACTORS.items():
        for lv in levels:
            out["%s=%s" % (f, lv)] = build(**{f: lv}, max_steps=max_steps)
    return out
