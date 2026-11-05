"""Task lifecycle state machine (benchmark profiles).

Retries and reassignment are separate on purpose: a reassignment (ASSIGNED -> READY because a
partner was released) is not a failure, and a FAILED task may be RETRIED only within an explicit
budget and with a recorded reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Tuple


class TaskState(str, Enum):
    PENDING = "pending"        # known, dependencies or arrival not yet satisfied
    READY = "ready"            # may be assigned
    ASSIGNED = "assigned"      # a crew is committed, execution not started
    ACTIVE = "active"          # physically being served (loaded / handled)
    SUSPENDED = "suspended"    # paused (e.g. crew member left to charge); resumes to ASSIGNED/ACTIVE
    FAILED = "failed"          # attempt failed (reason recorded); may be retried within budget
    COMPLETED = "completed"
    CANCELLED = "cancelled"


S = TaskState
ALLOWED_TRANSITIONS: Dict[TaskState, FrozenSet[TaskState]] = {
    S.PENDING: frozenset({S.READY, S.CANCELLED}),
    S.READY: frozenset({S.ASSIGNED, S.CANCELLED}),
    S.ASSIGNED: frozenset({S.ACTIVE, S.READY, S.SUSPENDED, S.FAILED, S.CANCELLED}),   # READY = reassignment
    S.ACTIVE: frozenset({S.COMPLETED, S.SUSPENDED, S.FAILED}),
    S.SUSPENDED: frozenset({S.ASSIGNED, S.ACTIVE, S.FAILED, S.CANCELLED}),
    S.FAILED: frozenset({S.READY, S.CANCELLED}),                                        # READY = retry
    S.COMPLETED: frozenset(),
    S.CANCELLED: frozenset(),
}
TERMINAL_STATES = frozenset(s for s, nxt in ALLOWED_TRANSITIONS.items() if not nxt)
REASON_REQUIRED = frozenset({S.SUSPENDED, S.FAILED, S.CANCELLED})


class InvalidTransition(ValueError):
    pass


@dataclass
class TaskRecord:
    task_id: int
    state: TaskState = TaskState.PENDING
    retry_budget: int = 0
    retries: int = 0
    reassignments: int = 0
    history: List[Tuple[int, TaskState, TaskState, Optional[str]]] = field(default_factory=list)

    def transition(self, new: TaskState, tick: int, reason: Optional[str] = None) -> None:
        new = TaskState(new)
        if new not in ALLOWED_TRANSITIONS[self.state]:
            raise InvalidTransition("task %d: %s -> %s not allowed" % (self.task_id, self.state.value, new.value))
        if new in REASON_REQUIRED and not reason:
            raise InvalidTransition("task %d: %s requires a reason" % (self.task_id, new.value))
        if self.history and tick < self.history[-1][0]:
            raise InvalidTransition("task %d: tick %d before previous transition" % (self.task_id, tick))
        if self.state == S.FAILED and new == S.READY:
            if self.retries >= self.retry_budget:
                raise InvalidTransition("task %d: retry budget %d exhausted" % (self.task_id, self.retry_budget))
            self.retries += 1
        if self.state == S.ASSIGNED and new == S.READY:
            self.reassignments += 1
        self.history.append((tick, self.state, new, reason))
        self.state = new

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES
