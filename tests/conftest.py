import copy

import pytest

from battery_tarware import ScenarioConfig, Simulator

TINY_MAP = """
C.....C
.......
.SS.SS.
.SS.SS.
.......
..G.G..
"""

AGV = {"name": "agv", "capabilities": ["transport"], "payload_kg": 20.0,
       "zones": ["aisle", "storage", "goal", "dock"],
       "energy": {"capacity_wh": 1000.0, "idle_w": 36.0, "move_w": 72.0, "payload_w_per_kg": 3.6,
                  "handling_w": 36.0, "max_charge_w": 720.0, "charge_efficiency": 1.0}}
PICKER = {"name": "picker", "capabilities": ["pick"], "zones": ["aisle", "goal", "dock"],
          "energy": {"capacity_wh": 500.0, "idle_w": 36.0, "move_w": 72.0, "payload_w_per_kg": 0.0,
                     "handling_w": 36.0, "max_charge_w": 720.0, "charge_efficiency": 1.0}}


def retrieval(name="job", pickers=1, weight=5.0, durations=(2, 1, 2)):
    roles = [{"name": "carrier", "capability": "transport", "presence": "on_cell"}]
    crew = ["carrier"]
    if pickers:
        roles.append({"name": "picker", "capability": "pick", "count": pickers})
        crew = ["carrier", "picker"]
    return {"name": name, "roles": roles, "carrier_role": "carrier", "weight": weight,
            "stages": [{"name": "pick", "at": "origin", "roles": crew, "duration": durations[0]},
                       {"name": "deliver", "at": "destination", "roles": ["carrier"], "duration": durations[1]},
                       {"name": "return", "at": "origin", "roles": crew, "duration": durations[2]}]}


def tiny_dict(**over):
    d = {
        "name": "tiny",
        "map": {"ascii": TINY_MAP},
        "robot_types": [copy.deepcopy(AGV), copy.deepcopy(PICKER)],
        "fleet": [{"type": "agv", "count": 2}, {"type": "picker", "count": 2}],
        "charger_types": [{"name": "standard", "power_w": 720.0}],
        "task_types": [retrieval()],
        "demand": [{"kind": "batch", "task_type": "job", "count": 1}],
        "seconds_per_tick": 5.0,
        "max_steps": 500,
    }
    d.update(over)
    return d


@pytest.fixture
def tiny():
    def make(**over):
        return Simulator(ScenarioConfig.from_dict(tiny_dict(**over)))
    return make


def place(sim, robot_id, cell):
    sim.robots[robot_id].cell = cell
    sim.robots[robot_id].clear_path()


def run(sim, ticks, actions=None):
    outs = []
    for _ in range(ticks):
        outs.append(sim.step(actions if actions is not None else [0] * len(sim.robots)))
    return outs
