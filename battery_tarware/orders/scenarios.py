"""Order-layer example scenarios (not registered as benchmark tiers; the built-in tiers are frozen).

``order_scenario()`` is the ``basic`` warehouse with an ``orders:`` section instead of demand
streams: outbound and inbound orders bring a shelf to a goal cell and back (crew by shelf weight,
as basic's solo / standard / large retrievals), relocations move a shelf to another rack location
(one AGV).
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


def order_scenario(source: Optional[Dict[str, Any]] = None, inventory: Optional[Dict[str, Any]] = None,
                   lifecycle: Optional[Dict[str, Any]] = None, base: str = "basic", **overrides) -> Dict[str, Any]:
    cfg = copy.deepcopy(get_scenario(base))
    cfg["name"] = "%s_orders" % base
    cfg["description"] = "%s warehouse driven by inbound / outbound / relocation orders" % base
    cfg["task_types"] = list(cfg["task_types"]) + [copy.deepcopy(RELOCATE)]
    cfg["demand"] = []
    cfg["orders"] = {
        "operations": {"outbound": {"task_types": BY_WEIGHT, "station": "goal"},
                       "inbound": {"task_types": BY_WEIGHT, "station": "goal"},
                       "relocation": {"task_types": [{"task_type": "relocate"}]}},
        "inventory": {"shelf_fill": 0.85, "package_fill": 0.6, "slot_capacity": 1,
                      "package_weight": {"kind": "uniform", "low": 0.5, "high": 16.0}, **(inventory or {})},
        "source": source or {"kind": "poisson", "rate": 0.02, "max_orders": 40,
                             "mix": {"outbound": 0.6, "inbound": 0.25, "relocation": 0.15},
                             "due_after": {"kind": "uniform", "low": 300, "high": 600}},
        "lifecycle": dict(lifecycle or {}),
    }
    cfg.update(overrides)
    return cfg
