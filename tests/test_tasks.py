"""Task dependency graphs and the lifecycle state machine."""
from types import SimpleNamespace

import pytest

from battery_tarware.tasks import (
    ALLOWED_TRANSITIONS, TERMINAL_STATES, InvalidTransition, TaskGraph, TaskGraphError, TaskRecord, TaskState,
)


def t(i, deps=(), required=None):
    return SimpleNamespace(id=i, depends_on=tuple(deps), required=required or {"transport": 1})


def test_graph_order_ready_and_descendants():
    g = TaskGraph([t(3), t(1), t(2, (1, 3)), t(4, (2,))])
    assert g.topological_order == (1, 3, 2, 4)
    assert g.ready(set()) == [1, 3] and g.ready({1, 3}) == [2]
    assert g.descendants(1) == {2, 4}


def test_graph_errors():
    with pytest.raises(TaskGraphError, match="cycle"):
        TaskGraph([t(1, (2,)), t(2, (1,))])
    with pytest.raises(TaskGraphError, match="unknown"):
        TaskGraph([t(1, (9,))])
    with pytest.raises(TaskGraphError, match="more robots"):
        TaskGraph([t(1, required={"pick": 3})], fleet={"pick": 2})


def test_lifecycle_rules():
    r = TaskRecord(1, retry_budget=1)
    r.transition(TaskState.READY, 0)
    r.transition(TaskState.ASSIGNED, 1)
    r.transition(TaskState.READY, 2)                          # reassignment, not failure
    assert r.reassignments == 1
    r.transition(TaskState.ASSIGNED, 3)
    r.transition(TaskState.ACTIVE, 4)
    with pytest.raises(InvalidTransition, match="reason"):
        r.transition(TaskState.FAILED, 5)
    r.transition(TaskState.FAILED, 5, reason="crew battery")
    r.transition(TaskState.READY, 6)                          # retry 1 of 1
    r.transition(TaskState.ASSIGNED, 7)
    r.transition(TaskState.FAILED, 8, reason="blocked")
    with pytest.raises(InvalidTransition, match="budget"):
        r.transition(TaskState.READY, 9)
    with pytest.raises(InvalidTransition):
        TaskRecord(2).transition(TaskState.COMPLETED, 0)
    assert TERMINAL_STATES == {TaskState.COMPLETED, TaskState.CANCELLED}
    assert set(ALLOWED_TRANSITIONS) == set(TaskState)
