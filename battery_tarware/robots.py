"""Robot instances: a resolved RobotTypeSpec plus mutable state."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from battery_tarware.config import RobotTypeSpec

Cell = Tuple[int, int]


class RobotStatus(str, Enum):
    IDLE = "idle"              # no commitment
    TASK = "task"              # committed to a task (travelling, waiting or working)
    CHARGE = "charge"          # heading to / docked at a charger
    FAILED = "failed"          # broken down, repairing
    STRANDED = "stranded"      # battery empty: permanently out of service


@dataclass
class Robot:
    id: int
    name: str
    spec: RobotTypeSpec
    type_index: int
    cell: Cell
    energy_wh: float
    status: RobotStatus = RobotStatus.IDLE
    task_id: Optional[int] = None
    role: Optional[str] = None
    charger: Optional[int] = None              # target / occupied charger id
    charging: bool = False
    path: List[Cell] = field(default_factory=list)
    path_goal: Optional[Cell] = None
    move_credit: float = 0.0
    blocked_ticks: int = 0
    elapsed: int = 0                               # ticks travelling towards the current target (PIBT priority)
    repair_left: int = 0
    resume_status: Optional[RobotStatus] = None   # status to return to after repair
    moved: bool = False                            # moved this tick (energy accounting)
    worked: bool = False                           # worked on a stage this tick
    energy_used_wh: float = 0.0
    distance: int = 0

    @property
    def soc(self) -> float:
        return self.energy_wh / self.spec.energy.capacity_wh

    @property
    def operational(self) -> bool:
        return self.status not in (RobotStatus.FAILED, RobotStatus.STRANDED)

    @property
    def capabilities(self) -> Tuple[str, ...]:
        return self.spec.capabilities

    def can(self, capability: str) -> bool:
        return capability in self.spec.capabilities

    def zones(self, carrying: bool) -> Tuple[str, ...]:
        if carrying and self.spec.zones_loaded is not None:
            return self.spec.zones_loaded
        return self.spec.zones

    def compatible_with(self, charger_type: str) -> bool:
        return self.spec.charger_types is None or charger_type in self.spec.charger_types

    def clear_path(self) -> None:
        self.path = []
        self.path_goal = None
        self.blocked_ticks = 0
