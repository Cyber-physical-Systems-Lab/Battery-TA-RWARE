"""Macro-mode movement: PIBT (Priority Inheritance with Backtracking).

Okumura, Machida, Défago, Tamura, "Priority inheritance with backtracking for iterative multi-agent
path finding", Artificial Intelligence 310 (2022). Each tick every movable unit (a robot, or a joint
carrier group moving as one) takes at most one step:

* units are processed by priority (ticks spent travelling towards the current target; ties by id);
* a unit tries its neighbouring cells (and staying) in order of remaining distance to its target;
* if the chosen cell is occupied, the occupant *inherits* the priority and must step away first
  (recursively); if it cannot, the unit backtracks to its next candidate;
* a pushed unit may not move into its pusher's cell;
* a restricted *swap* operation, after Okumura, "Improving LaCAM for scalable eventually optimal
  multi-agent pathfinding", IJCAI 2023, applies only when the wanted cell holds a unit trapped in a
  dead end;
* ties in distance prefer unoccupied cells (a choice made for this package).

Model 2.3: a joint carrier pair is a *rigid two-cell unit* (a convoy): its two robots occupy two
adjacent cells and move like a snake. One tick, either end may lead: the leading end steps into a
free neighbouring cell and the other end moves into the cell the leading end left, so a convoy can
also back out of a dead end (a shelf bay) through the cell its follower stood on. Only the one newly
entered cell has to be free; the pushed-occupant rules of PIBT apply to it unchanged.

The code is an independent implementation written for this package; no code was copied from the
reference implementations.

Distances are exact grid distances within the unit's allowed zones that respect one-way streets
(reverse BFS from the target, cached). Units that cannot move this tick (failed, charging, working a
stage, or without enough speed credit) are fixed obstacles.

Model 2.1: long-lived obstacles (charging, failed or stranded robots, and robots parked on their own
stage cell waiting for their crew or for their stage to finish) are walls in the distance field,
so a unit routes around them instead of waiting in front of them until they leave. Short-lived ones
(a stage being worked, a unit without speed credit) are still waited for. If no way around exists
the unit falls back to the model 2.0 field and waits.
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
    __slots__ = ("id", "members", "cell", "goal", "zones", "key", "fixed", "priority", "next", "lasting", "meet",
                 "cells")

    def __init__(self, id_, members, cell, goal, zones, key, fixed, priority, lasting=False, meet=None, cells=None):
        self.id, self.members, self.cell, self.goal = id_, members, cell, goal
        # cells the unit occupies: one for a robot, two (in member order) for a model 2.3 convoy
        self.cells: Tuple[Cell, ...] = tuple(cells) if cells is not None else (cell,)
        self.zones, self.key, self.fixed, self.priority = zones, key, fixed, priority
        self.lasting = lasting                   # model 2.1: fixed for longer than a tick or two
        self.meet = meet                         # model 2.1: (task id, cell) where this crew may share a cell
        self.next: Optional[Cell] = None


class PIBT:
    def __init__(self, sim):
        self.sim = sim
        self._dist: Dict[Tuple[Tuple[str, ...], Cell], np.ndarray] = {}
        self._walls: Tuple[Cell, ...] = ()
        self._wdist: Dict[Tuple, np.ndarray] = {}

    def route_distances(self, u: "Unit") -> np.ndarray:
        """Distances used to rank u's moves: model 2.1 treats long-lived obstacles as walls."""
        static = self.distances(u.zones, u.goal)
        walls = [c for c in self._walls if c != u.goal and c not in u.cells]
        if not walls or self.sim.config.model == "2.0":
            return static
        key = (u.zones, u.goal, tuple(walls))
        if key not in self._wdist:
            if len(self._wdist) > 4096:
                self._wdist.clear()
            pas = self.passable(u.zones).copy()
            for c in walls:
                pas[c] = False
            self._wdist[key] = reverse_distances(self.sim.layout, pas, u.goal)
        d = self._wdist[key]
        return d if min(d[c] for c in u.cells) != INF else static     # no way around: wait as in model 2.0

    def passable(self, zones: Tuple[str, ...]) -> np.ndarray:
        """Cells a unit may route through. Model 2.1: docks are destinations only (a unit enters a
        dock only as its goal), so no robot is routed or pushed through, or into, a dock bay."""
        pas = self.sim.passable(zones)
        cfg = getattr(self.sim, "config", None)
        if cfg is None or cfg.model == "2.0" or "dock" not in self.sim.layout.zones:
            return pas
        key = ("route",) + tuple(zones)
        cache = self.__dict__.setdefault("_route_pas", {})
        if key not in cache:
            cache[key] = pas & ~self.sim.layout.passable_mask(("dock",))
        return cache[key]

    def distances(self, zones: Tuple[str, ...], goal: Cell) -> np.ndarray:
        key = (zones, goal)
        if key not in self._dist:
            if len(self._dist) > 4096:
                self._dist.clear()
            self._dist[key] = reverse_distances(self.sim.layout, self.passable(zones), goal)
        return self._dist[key]

    def step(self, units: List[Unit]) -> None:
        self.reserved: Dict[Cell, List[Unit]] = {}
        self.now: Dict[Cell, List[Unit]] = {}
        for u in units:
            for c in u.cells:
                self.now.setdefault(c, []).append(u)
        for u in units:
            if u.fixed:
                self._reserve(u, u.cell if len(u.cells) == 1 else u.cells)
        self._walls = tuple(sorted({c for u in units if u.lasting for c in u.cells}))
        for u in sorted(units, key=lambda u: (-u.priority, u.id)):
            if u.next is None:
                self._pibt(u, None)

    def _compatible(self, a: Unit, b: Unit, cell: Optional[Cell] = None) -> bool:
        if a.key is not None and a.key == b.key:
            return True
        # model 2.1: a task's carrier and its on_cell crew may share the cell where they work together
        return a.meet is not None and a.meet == b.meet and cell == a.meet[1]

    def _reserve(self, u: Unit, c) -> None:
        """Reserve a cell (a robot) or the tuple of new cells (a convoy)."""
        u.next = c
        for x in ((c,) if len(u.cells) == 1 else c):
            self.reserved.setdefault(x, []).append(u)

    def _unreserve(self, u: Unit, c) -> None:
        for x in ((c,) if len(u.cells) == 1 else c):
            self.reserved[x].remove(u)
        u.next = None

    @staticmethod
    def _stays(w: Unit, c: Cell) -> bool:
        """Has ``w`` decided to occupy ``c`` next tick?"""
        return c in w.next if len(w.cells) == 2 else w.next == c

    def _candidates(self, u: Unit, parent: Optional[Unit] = None) -> List[Cell]:
        sim = self.sim
        pas = self.passable(u.zones)
        # model 2.1: a unit pushed by ``parent`` prefers stepping aside to retreating straight ahead
        # of it (equal distance), so a slow robot is not driven down a corridor by a faster one
        ahead = None
        if parent is not None and getattr(sim, "config", None) is not None and sim.config.model != "2.0":
            ahead = (2 * u.cell[0] - parent.cell[0], 2 * u.cell[1] - parent.cell[1])
        if u.goal is None:
            out = [(0, 0, 0, -1, u.cell)]
            for k, (dy, dx) in enumerate(MOVES):
                c = (u.cell[0] + dy, u.cell[1] + dx)
                if sim.layout.in_bounds(c) and pas[c] and sim.layout.move_ok(u.cell, c, k):
                    busy = any(w is not u and not self._compatible(u, w, c) for w in self.now.get(c, []))
                    out.append((1, int(c == ahead), int(busy), k, c))
            return [c for *_, c in sorted(out)]
        dist = self.route_distances(u)
        out = [(int(dist[u.cell]), 0, 0, -1, u.cell)]
        for k, (dy, dx) in enumerate(MOVES):
            c = (u.cell[0] + dy, u.cell[1] + dx)
            if not sim.layout.in_bounds(c) or not sim.layout.move_ok(u.cell, c, k):
                continue
            if pas[c] or c == u.goal:
                # ties in distance go to cells nobody occupies (avoids needless pushing back and forth)
                busy = any(w is not u and not self._compatible(u, w, c) for w in self.now.get(c, []))
                out.append((int(dist[c]), int(c == ahead), int(busy), k, c))
        return [c for *_, c in sorted(out)]

    def _dead_end(self, w: Unit, exit_cell: Cell) -> bool:
        """``w`` can only leave its cell through ``exit_cell`` (e.g. a loaded AGV in a storage bay)."""
        sim = self.sim
        pas = self.passable(w.zones)
        for k, (dy, dx) in enumerate(MOVES):
            c = (w.cell[0] + dy, w.cell[1] + dx)
            if c != exit_cell and sim.layout.in_bounds(c) and (pas[c] or c == w.goal) \
                    and sim.layout.move_ok(w.cell, c, k):
                return False
        return True

    def _occupant(self, u: Unit, c: Cell) -> Optional[Unit]:
        for w in self.now.get(c, []):
            if w is not u and not self._compatible(u, w, c) and w.next is None and not w.fixed:
                return w
        return None

    def _pibt(self, u: Unit, parent: Optional[Unit]) -> bool:
        if len(u.cells) == 2:
            return self._try_rigid(u, parent, self._moves_rigid(u, parent))
        cands = self._candidates(u, parent)
        # swap operation (Okumura, IJCAI 2023), restricted to dead ends: if the best cell holds a unit stuck in a dead end whose only
        # way out is u's cell, waiting cannot help -- u steps away instead and pulls that unit out
        swap = None
        if parent is None and cands and cands[0] != u.cell:
            w = self._occupant(u, cands[0])
            if w is not None and len(w.cells) == 1 and w.goal is not None and self._dead_end(w, u.cell):
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
            if any(not self._compatible(u, o, c) for o in self.reserved.get(c, []) if o is not u):
                continue
            self._reserve(u, c)
            ok = True
            for w in list(self.now.get(c, [])):
                if w is u or self._compatible(u, w, c) or w.next is not None:
                    if w is not u and not self._compatible(u, w, c) and self._stays(w, c):
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

    # ---- model 2.3: rigid two-cell convoy
    def _moves_rigid(self, u: Unit, parent: Optional[Unit]) -> List[Tuple[Cell, Cell]]:
        """The convoy's candidate placements ``(cell of member 0, cell of member 1)``, best first:
        staying, or either end leading into a free neighbour while the other end takes the cell the
        leading end left. Ranked by the remaining distance of the cell the leading end enters."""
        sim = self.sim
        pas = self.passable(u.zones)
        dist = self.route_distances(u) if u.goal is not None else None
        a, b = u.cells
        pcells = parent.cells if parent is not None else ()
        out = []
        stay_d = 0 if dist is None else int(min(dist[a], dist[b]))
        out.append((stay_d, 0, 0, -1, -1, (a, b)))
        for lead, (me, other) in enumerate(((a, b), (b, a))):
            for k, (dy, dx) in enumerate(MOVES):
                c = (me[0] + dy, me[1] + dx)
                if c == other or c in pcells or not sim.layout.in_bounds(c) or not sim.layout.move_ok(me, c, k):
                    continue
                if not (pas[c] or c == u.goal):
                    continue
                busy = any(w is not u and not self._compatible(u, w, c) for w in self.now.get(c, []))
                new = (c, a) if lead == 0 else (b, c)      # member 0 / member 1 after the move
                out.append((0 if dist is None else int(dist[c]), int(busy), 1, lead, k, new))
        out.sort(key=lambda m: m[:5])
        return [m[5] for m in out]

    def _try_rigid(self, u: Unit, parent: Optional[Unit], moves: List[Tuple[Cell, Cell]]) -> bool:
        for new in moves:
            if any(c in (parent.cells if parent is not None else ()) for c in new):
                continue                                         # no swapping with the pusher
            if any(not self._compatible(u, o, c) for c in new for o in self.reserved.get(c, []) if o is not u):
                continue
            self._reserve(u, new)
            ok = True
            for c in new:
                if c in u.cells:
                    continue                                     # a cell the convoy already holds
                for w in list(self.now.get(c, [])):
                    if w is u or self._compatible(u, w, c) or w.next is not None:
                        if w is not u and not self._compatible(u, w, c) and self._stays(w, c):
                            ok = False
                            break
                        continue
                    if w.fixed or not self._pibt(w, u):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
            self._unreserve(u, new)
        if u.next is None:
            self._reserve(u, u.cells)
        return False
