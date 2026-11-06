"""A task instance during a run: crew assignment, stage progress, payload location."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from battery_tarware.config import TaskTypeSpec
from battery_tarware.tasks.lifecycle import TaskRecord, TaskState

Cell = Tuple[int, int]


@dataclass
class Task:
    id: int
    type: TaskTypeSpec
    origin: Cell
    destination: Cell
    weight_kg: float
    released: int                                   # tick it became known
    stage_cells: Tuple[Cell, ...]                   # resolved location of every stage
    depends_on: Tuple[int, ...] = ()
    deadline: Optional[int] = None                  # absolute tick
    job_id: Optional[int] = None
    record: TaskRecord = None                       # lifecycle (set in __post_init__)
    assigned: Dict[str, List[int]] = field(default_factory=dict)   # role -> robot ids
    stage: int = 0                                  # current stage index
    progress: float = 0.0                           # work done on the current stage (ticks x rate)
    payload_cell: Optional[Cell] = None             # where the payload rests (None while carried)
    carried: bool = False
    move_credit: float = 0.0                        # joint-carrier speed accumulator
    completed_at: Optional[int] = None

    def __post_init__(self):
        if self.record is None:
            self.record = TaskRecord(self.id)
        for r in self.type.roles:
            self.assigned.setdefault(r.name, [])
        if self.type.carrier_role is not None and self.payload_cell is None and not self.carried:
            self.payload_cell = self.stage_cells[0]

    # ---- state
    @property
    def state(self) -> TaskState:
        return self.record.state

    @property
    def done(self) -> bool:
        return self.record.terminal

    @property
    def required(self) -> Dict[str, int]:
        req: Dict[str, int] = {}
        for r in self.type.roles:
            for c in r.capabilities:                    # an alternative slot counts under each capability
                req[c] = req.get(c, 0) + r.count
        return req

    def robots(self) -> List[int]:
        return [rid for ids in self.assigned.values() for rid in ids]

    def role_of(self, robot_id: int) -> Optional[str]:
        for role, ids in self.assigned.items():
            if robot_id in ids:
                return role
        return None

    def open_roles(self) -> Dict[str, int]:
        return {r.name: r.count - len(self.assigned[r.name]) for r in self.type.roles
                if len(self.assigned[r.name]) < r.count}

    def joinable_roles(self) -> Dict[str, int]:
        """Open roles that the *current* stage needs (crews are recruited stage by stage)."""
        need = set(self.type.stages[self.stage].roles) if self.stage < len(self.type.stages) else set()
        return {k: n for k, n in self.open_roles().items() if k in need}

    @property
    def fully_staffed(self) -> bool:
        return not self.open_roles()

    @property
    def carriers(self) -> List[int]:
        return list(self.assigned.get(self.type.carrier_role, [])) if self.type.carrier_role else []

    @property
    def stage_spec(self):
        return self.type.stages[self.stage]

    @property
    def stage_cell(self) -> Cell:
        return self.stage_cells[self.stage]

    def needs_role_at_stage(self, role: str, stage: Optional[int] = None) -> bool:
        s = self.type.stages[self.stage if stage is None else stage]
        return role in s.roles

    def next_stage_for(self, role: str) -> Optional[int]:
        for i in range(self.stage, len(self.type.stages)):
            if role in self.type.stages[i].roles:
                return i
        return None

    def age(self, tick: int) -> int:
        return tick - self.released
