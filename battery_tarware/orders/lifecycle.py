"""Extended task lifecycle of the order layer.

An order-layer task (``OrderTask``) lives across dispatch attempts; each attempt is one simulator
task. The simulator's own state machine (``battery_tarware.tasks.lifecycle``) is unchanged: while an
attempt runs, its transitions are mirrored here; ``pending``, ``blocked``, ``failed`` and
``retrying`` exist only at this layer.

=========  ==================================================================================
pending    accepted with its order, readiness not evaluated yet
blocked    waiting for a dependency, an inventory condition (package not stored, no free slot or
           location, shelf in use), or access/feasibility; the reason is recorded. Never released
           to the simulator, so no robot can be assigned to it.
ready      released as a simulator task (one dispatch attempt); eligible for assignment
assigned   crew (partly) committed, work not started
executing  work started (simulator ``active``)
suspended  crew released after work started; progress kept (simulator ``suspended``)
failed     the attempt failed (reason, attempt number, blocking resource, last progress tick and
           a state snapshot are recorded); final when no retry is left
retrying   failed with retries left; waits ``retry_delay`` ticks, then readiness is rechecked
completed  the simulator task completed
cancelled  terminal; reason recorded
=========  ==================================================================================
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet


class OTaskState(str, Enum):
    PENDING = "pending"
    BLOCKED = "blocked"
    READY = "ready"
    ASSIGNED = "assigned"
    EXECUTING = "executing"
    SUSPENDED = "suspended"
    FAILED = "failed"
    RETRYING = "retrying"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


O = OTaskState
TRANSITIONS: Dict[OTaskState, FrozenSet[OTaskState]] = {
    O.PENDING: frozenset({O.BLOCKED, O.READY, O.FAILED, O.CANCELLED}),
    O.BLOCKED: frozenset({O.BLOCKED, O.READY, O.FAILED, O.CANCELLED}),     # BLOCKED -> BLOCKED: new reason
    O.READY: frozenset({O.ASSIGNED, O.FAILED, O.CANCELLED}),
    O.ASSIGNED: frozenset({O.EXECUTING, O.READY, O.SUSPENDED, O.FAILED, O.CANCELLED}),   # READY = reassignment
    O.EXECUTING: frozenset({O.COMPLETED, O.SUSPENDED, O.FAILED, O.CANCELLED}),
    O.SUSPENDED: frozenset({O.ASSIGNED, O.EXECUTING, O.FAILED, O.CANCELLED}),
    O.FAILED: frozenset({O.RETRYING, O.CANCELLED}),
    O.RETRYING: frozenset({O.BLOCKED, O.READY, O.FAILED, O.CANCELLED}),
    O.COMPLETED: frozenset(),
    O.CANCELLED: frozenset(),
}
REASON_REQUIRED = frozenset({O.BLOCKED, O.SUSPENDED, O.FAILED, O.RETRYING, O.CANCELLED})
STARTED = frozenset({O.EXECUTING, O.SUSPENDED})        # physical work may have begun

# simulator state -> order-layer state while an attempt is attached
FROM_SIM = {"ready": O.READY, "assigned": O.ASSIGNED, "active": O.EXECUTING, "suspended": O.SUSPENDED,
            "completed": O.COMPLETED}


class InvalidOTransition(ValueError):
    pass


def check_transition(task_id: str, old: OTaskState, new: OTaskState, reason) -> None:
    if new not in TRANSITIONS[old]:
        raise InvalidOTransition("task %s: %s -> %s not allowed" % (task_id, old.value, new.value))
    if new in REASON_REQUIRED and not reason:
        raise InvalidOTransition("task %s: %s requires a reason" % (task_id, new.value))
