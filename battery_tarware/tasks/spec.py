"""TaskSpec: what a task needs, independent of how a coordinator serves it."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from battery_tarware.definitions import PACKAGE_REQUIREMENTS, PackageType


class Capability(str, Enum):
    """Robot capability classes. Values are stable strings (used in configs and DTOs)."""
    TRANSPORT = "transport"   # AGV: carries a shelf
    PICK = "pick"             # picker: load / unload assist


class TaskKind(str, Enum):
    """Task families. ``RETRIEVAL`` is the only kind the baseline kernel executes today
    (shelf -> goal -> shelf return). The others are benchmark-profile kinds whose execution rules
    are not implemented yet; specs of those kinds are valid data but must not be fed to the
    legacy/protocol_v4 kernel."""
    RETRIEVAL = "retrieval"
    INBOUND = "inbound"
    RELOCATION = "relocation"
    COOPERATIVE_HANDLING = "cooperative_handling"


Cell = Tuple[int, int]   # (y, x), contracts v1 section 3


@dataclass(frozen=True)
class TaskSpec:
    id: int
    kind: TaskKind
    required: Mapping[Capability, int]
    origin: Optional[Cell] = None
    destination: Optional[Cell] = None
    arrival_tick: int = 0
    deadline_tick: Optional[int] = None
    depends_on: Tuple[int, ...] = ()
    priority: int = 0
    package_type: Optional[PackageType] = None   # set when derived from a baseline package

    def __post_init__(self):
        req = {Capability(k): int(v) for k, v in dict(self.required).items()}
        if any(v < 0 for v in req.values()):
            raise ValueError("task %d: negative capability count %r" % (self.id, req))
        if sum(req.values()) == 0:
            raise ValueError("task %d: requires no robot at all" % self.id)
        object.__setattr__(self, "required", MappingProxyType(req))
        object.__setattr__(self, "kind", TaskKind(self.kind))
        deps = tuple(int(d) for d in self.depends_on)
        if self.id in deps:
            raise ValueError("task %d depends on itself" % self.id)
        if len(set(deps)) != len(deps):
            raise ValueError("task %d: duplicate dependency in %r" % (self.id, deps))
        object.__setattr__(self, "depends_on", deps)
        if self.arrival_tick < 0:
            raise ValueError("task %d: negative arrival_tick" % self.id)
        if self.deadline_tick is not None and self.deadline_tick < self.arrival_tick:
            raise ValueError("task %d: deadline before arrival" % self.id)

    @property
    def crew_size(self) -> int:
        return sum(self.required.values())

    def count(self, cap: Capability) -> int:
        return self.required.get(cap, 0)


def requirements_from_package(package_type: PackageType) -> Mapping[Capability, int]:
    """Baseline package semantics: PACKAGE_REQUIREMENTS = (AGVs, pickers)."""
    agvs, pickers = PACKAGE_REQUIREMENTS[PackageType(package_type)]
    req = {Capability.TRANSPORT: agvs}
    if pickers:
        req[Capability.PICK] = pickers
    return req


def task_from_package(task_id: int, package_type: PackageType, origin: Cell,
                      arrival_tick: int = 0) -> TaskSpec:
    """The TaskSpec equivalent of one baseline request (a shelf in ``request_queue``)."""
    return TaskSpec(id=task_id, kind=TaskKind.RETRIEVAL, required=requirements_from_package(package_type),
                    origin=origin, arrival_tick=arrival_tick, package_type=PackageType(package_type))
