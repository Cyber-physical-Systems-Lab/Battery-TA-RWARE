"""Read-only state for coordinators, and a primitive-array DTO for C++ / other languages.

``StateView`` is a frozen value: building one draws no random numbers and mutates nothing.
``StateView.to_arrays()`` returns contiguous, fixed-dtype numpy arrays (schema ``SCHEMA_VERSION``)
that a native coordinator can read without touching Python objects (e.g. through pybind11's
``py::array_t``). Coordinates are ``(y, x)`` int32; ids are int32 and index the arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from battery_tarware.robots import RobotStatus

SCHEMA_VERSION = 1
STATUS_CODES = {s: i for i, s in enumerate(RobotStatus)}
TASK_STATE_CODES = {s: i for i, s in enumerate(
    ["pending", "ready", "assigned", "active", "suspended", "failed", "completed", "cancelled"])}

Cell = Tuple[int, int]


@dataclass(frozen=True)
class RobotView:
    id: int
    name: str
    type: str
    capabilities: Tuple[str, ...]
    cell: Cell
    soc: float
    energy_wh: float
    capacity_wh: float
    speed: float
    payload_kg: float
    handling_speed: float
    status: str
    task_id: Optional[int]
    role: Optional[str]
    charger: Optional[int]
    charging: bool
    carrying: bool


@dataclass(frozen=True)
class TaskView:
    id: int
    type: str
    state: str
    origin: Cell
    destination: Cell
    stage: int
    n_stages: int
    stage_cell: Cell
    weight_kg: float
    released: int
    deadline: Optional[int]
    depends_on: Tuple[int, ...]
    assigned: Tuple[Tuple[str, Tuple[int, ...]], ...]
    open_roles: Tuple[Tuple[str, str, int], ...]          # (role, capability, open slots)
    carried: bool
    payload_cell: Optional[Cell]


@dataclass(frozen=True)
class ChargerView:
    id: int
    cell: Cell
    type: str
    power_w: float
    occupant: Optional[int]
    reserved_by: Optional[int] = None      # model 2.1 reservation (None in model 2.0)


@dataclass(frozen=True)
class StateView:
    schema_version: int
    tick: int
    grid_shape: Tuple[int, int]
    robots: Tuple[RobotView, ...]
    tasks: Tuple[TaskView, ...]                           # every non-terminal task
    chargers: Tuple[ChargerView, ...]
    capabilities: Tuple[str, ...]

    def to_arrays(self) -> Dict[str, np.ndarray]:
        caps = {c: i for i, c in enumerate(self.capabilities)}
        R, T, C = len(self.robots), len(self.tasks), len(self.chargers)
        a = {
            "schema_version": np.array([self.schema_version], np.int32),
            "tick": np.array([self.tick], np.int64),
            "robot_cell": np.array([r.cell for r in self.robots], np.int32).reshape(R, 2),
            "robot_soc": np.array([r.soc for r in self.robots], np.float64),
            "robot_speed": np.array([r.speed for r in self.robots], np.float64),
            "robot_payload_kg": np.array([r.payload_kg for r in self.robots], np.float64),
            "robot_status": np.array([STATUS_CODES[RobotStatus(r.status)] for r in self.robots], np.int32),
            "robot_task": np.array([-1 if r.task_id is None else r.task_id for r in self.robots], np.int32),
            "robot_capabilities": np.array([[c in r.capabilities for c in self.capabilities] for r in self.robots],
                                           np.uint8).reshape(R, len(self.capabilities)),
            "task_id": np.array([t.id for t in self.tasks], np.int32),
            "task_state": np.array([TASK_STATE_CODES[t.state] for t in self.tasks], np.int32),
            "task_stage_cell": np.array([t.stage_cell for t in self.tasks], np.int32).reshape(T, 2),
            "task_weight_kg": np.array([t.weight_kg for t in self.tasks], np.float64),
            "task_open_capability": np.zeros((T, len(self.capabilities)), np.int32),
            "charger_cell": np.array([c.cell for c in self.chargers], np.int32).reshape(C, 2),
            "charger_power_w": np.array([c.power_w for c in self.chargers], np.float64),
            "charger_occupant": np.array([-1 if c.occupant is None else c.occupant for c in self.chargers], np.int32),
        }
        for i, t in enumerate(self.tasks):
            for _, cap, n in t.open_roles:
                a["task_open_capability"][i, caps[cap]] += n
        return {k: np.ascontiguousarray(v) for k, v in a.items()}


def build_state_view(sim) -> StateView:
    robots = tuple(RobotView(
        r.id, r.name, sim.type_names[r.type_index], r.spec.capabilities, r.cell, r.soc, r.energy_wh,
        r.spec.energy.capacity_wh, r.spec.speed, r.spec.payload_kg, r.spec.handling_speed, r.status.value,
        r.task_id, r.role, r.charger, r.charging, sim.carrying(r)) for r in sim.robots)
    tasks = tuple(TaskView(
        t.id, t.type.name, t.state.value, t.origin, t.destination, t.stage, len(t.type.stages),
        t.stage_cell, t.weight_kg, t.released, t.deadline, t.depends_on,
        tuple((k, tuple(v)) for k, v in t.assigned.items()),
        tuple((role, t.type.role(role).capability, n) for role, n in t.open_roles().items()),
        t.carried, t.payload_cell) for t in sim.tasks.values() if not t.done)
    chargers = tuple(ChargerView(c.id, c.cell, c.type, c.power_w, c.occupant, c.reserved_by) for c in sim.chargers)
    return StateView(SCHEMA_VERSION, sim.tick, sim.layout.shape, robots, tasks, chargers, tuple(sim.capabilities))
