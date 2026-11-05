"""Macro-mode movement: PIBT (Priority Inheritance with Backtracking).

Okumura, Machida, Défago, Tamura, "Priority inheritance with backtracking for iterative multi-agent
path finding", Artificial Intelligence 310 (2022). Each tick every movable unit (a robot, or a joint
carrier group moving as one) takes at most one step:

* units are processed by priority (ticks spent travelling towards the current target; ties by id);
* a unit tries its neighbouring cells (and staying) in order of remaining distance to its target;
* if the chosen cell is occupied, the occupant *inherits* the priority and must step away first
  (recursively); if it cannot, the unit backtracks to its next candidate;
* a pushed unit may not move into its pusher's cell (no swaps).

Distances are exact grid distances within the unit's allowed zones that respect one-way streets
(reverse BFS from the target, cached). Units that cannot move this tick (failed, charging, working a
stage, or without enough speed credit) are fixed obstacles.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from battery_tarware.navigation import MOVES

Cell = Tuple[int, int]
INF = np.iinfo(np.int32).max


def reverse_distances(layout, passable: np.ndarray, goal: Cell) -> np.ndarray:
    """dist[c] = steps from c to goal through passable cells (goal itself may be impassable)."""
    H, W = passable.shape
    dist = np.full((H, W), INF, dtype=np.int32)
    dist[goal] = 0
    q = deque([goal])
    while q:
        v = q.popleft()
        d = dist[v] + 1
        for k, (dy, dx) in enumerate(MOVES):
            u = (v[0] - dy, v[1] - dx)                 # u --k--> v
            if 0 <= u[0] < H and 0 <= u[1] < W and dist[u] == INF and passable[u] and layout.move_ok(u, v, k):
                dist[u] = d
                q.append(u)
    return dist


class Unit:
    __slots__ = ("id", "members", "cell", "goal", "zones", "key", "fixed", "priority", "next")

    def __init__(self, id_, members, cell, goal, zones, key, fixed, priority):
        self.id, self.members, self.cell, self.goal = id_, members, cell, goal
        self.zones, self.key, self.fixed, self.priority = zones, key, fixed, priority
        self.next: Optional[Cell] = None


class PIBT:
    def __init__(self, sim):
        self.sim = sim
        self._dist: Dict[Tuple[Tuple[str, ...], Cell], np.ndarray] = {}

    def distances(self, zones: Tuple[str, ...], goal: Cell) -> np.ndarray:
        key = (zones, goal)
        if key not in self._dist:
            if len(self._dist) > 4096:
                self._dist.clear()
            self._dist[key] = reverse_distances(self.sim.layout, self.sim.passable(zones), goal)
        return self._dist[key]

    def step(self, units: List[Unit]) -> None:
        self.reserved: Dict[Cell, List[Unit]] = {}
        self.now: Dict[Cell, List[Unit]] = {}
        for u in units:
            self.now.setdefault(u.cell, []).append(u)
        for u in units:
            if u.fixed:
                self._reserve(u, u.cell)
        for u in sorted(units, key=lambda u: (-u.priority, u.id)):
            if u.next is None:
                self._pibt(u, None)

    def _compatible(self, a: Unit, b: Unit) -> bool:
        return a.key is not None and a.key == b.key

    def _reserve(self, u: Unit, c: Cell) -> None:
        u.next = c
        self.reserved.setdefault(c, []).append(u)

    def _unreserve(self, u: Unit, c: Cell) -> None:
        self.reserved[c].remove(u)
        u.next = None

    def _candidates(self, u: Unit) -> List[Cell]:
        sim = self.sim
        pas = sim.passable(u.zones)
        if u.goal is None:
            out = [(0, 0, -1, u.cell)]
            for k, (dy, dx) in enumerate(MOVES):
                c = (u.cell[0] + dy, u.cell[1] + dx)
                if sim.layout.in_bounds(c) and pas[c] and sim.layout.move_ok(u.cell, c, k):
                    busy = any(w is not u and not self._compatible(u, w) for w in self.now.get(c, []))
                    out.append((1, int(busy), k, c))
            return [c for *_, c in sorted(out)]
        dist = self.distances(u.zones, u.goal)
        out = [(int(dist[u.cell]), 0, -1, u.cell)]
        for k, (dy, dx) in enumerate(MOVES):
            c = (u.cell[0] + dy, u.cell[1] + dx)
            if not sim.layout.in_bounds(c) or not sim.layout.move_ok(u.cell, c, k):
                continue
            if pas[c] or c == u.goal:
                # ties in distance go to cells nobody occupies (avoids needless pushing back and forth)
                busy = any(w is not u and not self._compatible(u, w) for w in self.now.get(c, []))
                out.append((int(dist[c]), int(busy), k, c))
        return [c for *_, c in sorted(out)]

    def _dead_end(self, w: Unit, exit_cell: Cell) -> bool:
        """``w`` can only leave its cell through ``exit_cell`` (e.g. a loaded AGV in a storage bay)."""
        sim = self.sim
        pas = sim.passable(w.zones)
        for k, (dy, dx) in enumerate(MOVES):
            c = (w.cell[0] + dy, w.cell[1] + dx)
            if c != exit_cell and sim.layout.in_bounds(c) and (pas[c] or c == w.goal) \
                    and sim.layout.move_ok(w.cell, c, k):
                return False
        return True

    def _occupant(self, u: Unit, c: Cell) -> Optional[Unit]:
        for w in self.now.get(c, []):
            if w is not u and not self._compatible(u, w) and w.next is None and not w.fixed:
                return w
        return None

    def _pibt(self, u: Unit, parent: Optional[Unit]) -> bool:
        cands = self._candidates(u)
        # swap operation (Okumura, LaCAM3): if the best cell holds a unit stuck in a dead end whose only
        # way out is u's cell, waiting cannot help -- u steps away instead and pulls that unit out
        swap = None
        if parent is None and cands and cands[0] != u.cell:
            w = self._occupant(u, cands[0])
            if w is not None and w.goal is not None and self._dead_end(w, u.cell):
                wc = self._candidates(w)
                if wc and wc[0] == u.cell:
                    swap = w
                    cands = [c for c in cands[1:] if c != u.cell] + [cands[0]]
        ok = self._try(u, parent, cands)
        if ok and swap is not None and swap.next is None and u.next != u.cell:
            self._pibt(swap, None)
        return ok

    def _try(self, u: Unit, parent: Optional[Unit], cands: List[Cell]) -> bool:
        for c in cands:
            if parent is not None and c == parent.cell:
                continue                                         # no swapping with the pusher
            if any(not self._compatible(u, o) for o in self.reserved.get(c, []) if o is not u):
                continue
            self._reserve(u, c)
            ok = True
            for w in list(self.now.get(c, [])):
                if w is u or self._compatible(u, w) or w.next is not None:
                    if w is not u and not self._compatible(u, w) and w.next == c:
                        ok = False                               # occupant stays here
                        break
                    continue
                if w.fixed or not self._pibt(w, u):
                    ok = False
                    break
            if ok:
                return True
            self._unreserve(u, c)
        if u.next is None:
            self._reserve(u, u.cell)
        return False
