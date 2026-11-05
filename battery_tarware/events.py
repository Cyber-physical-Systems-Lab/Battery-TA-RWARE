"""Append-only event log of state transitions (what happened, not inferred afterwards)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

EVENT_TYPES = (
    "task_released", "task_ready", "task_cancelled", "task_completed", "task_suspended",
    "robot_joined", "robot_released", "stage_started", "stage_completed", "payload_picked",
    "payload_dropped", "payload_placed", "charge_started", "charge_stopped", "robot_failed",
    "robot_repaired", "robot_stranded", "collision_blocked",
)


@dataclass(frozen=True)
class Event:
    tick: int                      # tick at whose end the event happened (1-based after the first step)
    seq: int                       # order within the episode
    type: str
    robots: Tuple[int, ...] = ()
    task: Optional[int] = None
    data: Mapping[str, Any] = field(default_factory=dict)


class EventLog:
    def __init__(self):
        self.events: List[Event] = []

    def emit(self, tick: int, type_: str, robots=(), task: Optional[int] = None, **data) -> None:
        assert type_ in EVENT_TYPES, type_
        self.events.append(Event(tick, len(self.events), type_, tuple(robots), task, dict(data)))

    def since(self, cursor: int) -> List[Event]:
        return self.events[cursor:]

    def __len__(self) -> int:
        return len(self.events)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.events:
            out[e.type] = out.get(e.type, 0) + 1
        return out
