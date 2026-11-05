"""Track B: task model, dependency DAG and lifecycle (pure modules, no simulator wiring)."""
import pytest

from battery_tarware.definitions import PACKAGE_REQUIREMENTS, PackageType
from battery_tarware.tasks import (
    ALLOWED_TRANSITIONS, Capability, InvalidTransition, TaskGraph, TaskGraphError, TaskKind,
    TaskRecord, TaskSpec, TaskState, TERMINAL_STATES, requirements_from_package, task_from_package,
)

T, P = Capability.TRANSPORT, Capability.PICK


def spec(i, deps=(), req=None, **kw):
    return TaskSpec(id=i, kind=TaskKind.RETRIEVAL, required=req or {T: 1}, depends_on=deps, **kw)


@pytest.mark.parametrize("pkg", list(PackageType))
def test_package_requirements_round_trip(pkg):
    agvs, pickers = PACKAGE_REQUIREMENTS[pkg]
    t = task_from_package(7, pkg, origin=(3, 4))
    assert t.count(T) == agvs and t.count(P) == pickers
    assert t.crew_size == agvs + pickers and t.package_type == pkg and t.kind == TaskKind.RETRIEVAL


def test_spec_validation():
    with pytest.raises(ValueError):
        spec(1, req={T: 0})
    with pytest.raises(ValueError):
        spec(1, req={T: -1})
    with pytest.raises(ValueError):
        spec(1, deps=(1,))
    with pytest.raises(ValueError):
        spec(1, deps=(2, 2))
    with pytest.raises(ValueError):
        spec(1, arrival_tick=10, deadline_tick=5)
    t = spec(1, req={"transport": 1, "pick": 2})
    assert t.required == {T: 1, P: 2}
    with pytest.raises(TypeError):
        t.required[T] = 5                     # frozen mapping


def test_graph_order_is_deterministic_and_ready_set():
    g = TaskGraph([spec(3), spec(1), spec(2, deps=(1, 3)), spec(4, deps=(2,), arrival_tick=50)])
    assert g.topological_order == (1, 3, 2, 4)
    assert g.ready(completed=set()) == [1, 3]
    assert g.ready(completed={1}) == [3]
    assert g.ready(completed={1, 3}) == [2]
    assert g.ready(completed={1, 2, 3}, tick=10) == []
    assert g.ready(completed={1, 2, 3}, tick=50) == [4]
    assert g.descendants(1) == {2, 4}


def test_graph_rejects_cycles_unknown_deps_duplicates():
    with pytest.raises(TaskGraphError, match="cycle"):
        TaskGraph([spec(1, deps=(2,)), spec(2, deps=(1,))])
    with pytest.raises(TaskGraphError, match="unknown"):
        TaskGraph([spec(1, deps=(9,))])
    with pytest.raises(TaskGraphError, match="duplicate"):
        TaskGraph([spec(1), spec(1)])


def test_graph_rejects_capability_demand_beyond_fleet():
    tasks = [spec(1, req={T: 1, P: 2}), spec(2, req={T: 1})]
    TaskGraph(tasks, fleet={T: 3, P: 2})
    with pytest.raises(TaskGraphError, match="more robots"):
        TaskGraph(tasks, fleet={T: 3, P: 1})


def test_lifecycle_happy_path_and_terminal():
    r = TaskRecord(1)
    for i, s in enumerate([TaskState.READY, TaskState.ASSIGNED, TaskState.ACTIVE, TaskState.COMPLETED]):
        r.transition(s, tick=i)
    assert r.terminal and [h[2] for h in r.history][-1] == TaskState.COMPLETED
    with pytest.raises(InvalidTransition):
        r.transition(TaskState.READY, tick=9)


def test_reassignment_is_not_failure_and_retry_has_budget():
    r = TaskRecord(1, retry_budget=1)
    r.transition(TaskState.READY, 0); r.transition(TaskState.ASSIGNED, 1)
    r.transition(TaskState.READY, 2)                          # partner released: reassignment
    assert r.reassignments == 1 and r.retries == 0
    r.transition(TaskState.ASSIGNED, 3); r.transition(TaskState.ACTIVE, 4)
    with pytest.raises(InvalidTransition, match="reason"):
        r.transition(TaskState.FAILED, 5)
    r.transition(TaskState.FAILED, 5, reason="crew battery")
    r.transition(TaskState.READY, 6)                          # retry 1 of 1
    r.transition(TaskState.ASSIGNED, 7); r.transition(TaskState.FAILED, 8, reason="blocked")
    with pytest.raises(InvalidTransition, match="budget"):
        r.transition(TaskState.READY, 9)


def test_transition_table_is_closed():
    assert set(ALLOWED_TRANSITIONS) == set(TaskState)
    assert TERMINAL_STATES == {TaskState.COMPLETED, TaskState.CANCELLED}
    for nxt in ALLOWED_TRANSITIONS.values():
        assert nxt <= set(TaskState)


def test_tick_monotonic():
    r = TaskRecord(1)
    r.transition(TaskState.READY, 5)
    with pytest.raises(InvalidTransition):
        r.transition(TaskState.ASSIGNED, 4)


def test_legacy_profiles_do_not_import_tasks():
    import subprocess, sys
    code = ("import sys, battery_tarware, battery_tarware.protocol_v4; "
            "assert 'battery_tarware.tasks' not in sys.modules, 'kernel imports tasks'")
    flags = ["-S"] if sys.flags.no_site else []
    p = subprocess.run([sys.executable] + flags + ["-c", code], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
