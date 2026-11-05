"""Task model for Battery-TA-RWARE benchmark profiles (Track B, contracts v1 section 8).

Pure data + validation. Nothing here is wired into ``Warehouse.step``: the ``legacy`` and
``protocol_v4`` profiles do not import this package, so their G0 fingerprints cannot change.
"""
from battery_tarware.tasks.spec import (
    Capability, TaskKind, TaskSpec, requirements_from_package, task_from_package,
)
from battery_tarware.tasks.graph import TaskGraph, TaskGraphError
from battery_tarware.tasks.lifecycle import (
    TaskState, TaskRecord, InvalidTransition, ALLOWED_TRANSITIONS, TERMINAL_STATES,
)

__all__ = [
    "Capability", "TaskKind", "TaskSpec", "requirements_from_package", "task_from_package",
    "TaskGraph", "TaskGraphError",
    "TaskState", "TaskRecord", "InvalidTransition", "ALLOWED_TRANSITIONS", "TERMINAL_STATES",
]
