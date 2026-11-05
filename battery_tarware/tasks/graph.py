"""Task dependency DAG: validation, deterministic topological order, ready sets."""
from __future__ import annotations

import heapq
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

from battery_tarware.tasks.spec import Capability, TaskSpec


class TaskGraphError(ValueError):
    pass


class TaskGraph:
    """Immutable DAG over TaskSpecs. Tie-breaks are by task id, so every order is deterministic."""

    def __init__(self, tasks: Iterable[TaskSpec], fleet: Optional[Mapping[Capability, int]] = None):
        self._tasks: Dict[int, TaskSpec] = {}
        for t in tasks:
            if t.id in self._tasks:
                raise TaskGraphError("duplicate task id %d" % t.id)
            self._tasks[t.id] = t
        for t in self._tasks.values():
            missing = [d for d in t.depends_on if d not in self._tasks]
            if missing:
                raise TaskGraphError("task %d depends on unknown task(s) %r" % (t.id, missing))
        if fleet is not None:
            self._check_feasible(fleet)
        self._children: Dict[int, Tuple[int, ...]] = {i: () for i in self._tasks}
        for t in self._tasks.values():
            for d in t.depends_on:
                self._children[d] = self._children[d] + (t.id,)
        self._order = self._toposort()

    def _check_feasible(self, fleet: Mapping[Capability, int]) -> None:
        bad = []
        for t in self._tasks.values():
            short = {c.value: (n, fleet.get(c, 0)) for c, n in t.required.items() if n > fleet.get(c, 0)}
            if short:
                bad.append((t.id, short))
        if bad:
            raise TaskGraphError("tasks need more robots than the fleet has (needed, available): %r" % bad)

    def _toposort(self) -> Tuple[int, ...]:
        indeg = {i: len(t.depends_on) for i, t in self._tasks.items()}
        heap = [i for i, d in indeg.items() if d == 0]
        heapq.heapify(heap)
        order: List[int] = []
        while heap:
            i = heapq.heappop(heap)
            order.append(i)
            for c in self._children[i]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    heapq.heappush(heap, c)
        if len(order) != len(self._tasks):
            cyc = sorted(i for i, d in indeg.items() if d > 0)
            raise TaskGraphError("dependency cycle among tasks %r" % cyc)
        return tuple(order)

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, task_id: int) -> TaskSpec:
        return self._tasks[task_id]

    def __iter__(self):
        return (self._tasks[i] for i in self._order)

    @property
    def topological_order(self) -> Tuple[int, ...]:
        return self._order

    def children(self, task_id: int) -> Tuple[int, ...]:
        return self._children[task_id]

    def ready(self, completed: Set[int], released: Optional[Set[int]] = None, tick: Optional[int] = None) -> List[int]:
        """Tasks whose dependencies are all completed, not themselves completed / released, and
        (if ``tick`` is given) already arrived. Sorted by id."""
        released = released or set()
        out = []
        for i in self._order:
            t = self._tasks[i]
            if i in completed or i in released:
                continue
            if tick is not None and t.arrival_tick > tick:
                continue
            if all(d in completed for d in t.depends_on):
                out.append(i)
        return sorted(out)

    def descendants(self, task_id: int) -> Set[int]:
        seen: Set[int] = set()
        stack = list(self._children[task_id])
        while stack:
            c = stack.pop()
            if c not in seen:
                seen.add(c)
                stack.extend(self._children[c])
        return seen
