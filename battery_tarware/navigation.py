"""Path planning and simultaneous-move conflict resolution.

Robots plan 4-connected shortest paths (A*, Manhattan heuristic, deterministic tie-breaks) over the
cells their zone permissions allow. All movers then move at once; conflicts are resolved by a fixed
priority (lower mover id wins): a move fails if its target cell stays occupied, if a higher-priority
mover claims the same cell, or if two movers would swap cells. Chains (A follows B into the cell B
leaves) succeed.
"""
from __future__ import annotations

import heapq
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Cell = Tuple[int, int]
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))     # up, down, left, right
OPPOSITE = (1, 0, 3, 2)


def astar(passable: np.ndarray, start: Cell, goal: Cell, blocked: Optional[Set[Cell]] = None,
          directions: Optional[np.ndarray] = None) -> Optional[List[Cell]]:
    """Shortest path from ``start`` to ``goal`` (both included). ``start`` and ``goal`` need not be
    passable (robots may stand on, or head for, a cell outside their normal zones, e.g. a task cell).
    ``directions`` (from Layout) enforces one-way aisles. Returns None if unreachable."""
    if start == goal:
        return [start]
    H, W = passable.shape
    blocked = blocked or set()

    def ok(c: Cell) -> bool:
        return c == goal or (passable[c] and c not in blocked)

    g = {start: 0}
    parent: Dict[Cell, Cell] = {}
    h0 = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    heap = [(h0, 0, start)]
    while heap:
        f, gc, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        if gc > g.get(cur, 1 << 30):
            continue
        for k, (dy, dx) in enumerate(MOVES):
            n = (cur[0] + dy, cur[1] + dx)
            if not (0 <= n[0] < H and 0 <= n[1] < W) or not ok(n):
                continue
            if directions is not None:
                da, db = directions[cur], directions[n]
                if (da >= 0 and da != k and db != -1) or (db >= 0 and db == OPPOSITE[k]):
                    continue                     # one-way streets (see Layout.move_ok)
            ng = gc + 1
            if ng < g.get(n, 1 << 30):
                g[n] = ng
                parent[n] = cur
                heapq.heappush(heap, (ng + abs(n[0] - goal[0]) + abs(n[1] - goal[1]), ng, n))
    return None


def bfs_distances(passable: np.ndarray, start: Cell) -> np.ndarray:
    """Grid distance from ``start`` to every cell reachable through passable cells (-1 elsewhere),
    ignoring one-way directions (an optimistic estimate). Non-passable cells adjacent to the
    reachable region get a distance too (they can be targets)."""
    H, W = passable.shape
    dist = np.full((H, W), -1, dtype=np.int32)
    dist[start] = 0
    frontier = [start]
    while frontier:
        nxt = []
        for c in frontier:
            d = dist[c] + 1
            for dy, dx in MOVES:
                n = (c[0] + dy, c[1] + dx)
                if 0 <= n[0] < H and 0 <= n[1] < W and dist[n] < 0:
                    dist[n] = d
                    if passable[n]:
                        nxt.append(n)
        frontier = nxt
    return dist


def resolve_moves(positions: Dict[int, Cell], intents: Dict[int, Cell], shared: Dict[int, int]) -> Set[int]:
    """Decide which movers succeed.

    positions  mover id -> current cell (every mover, moving or not)
    intents    mover id -> desired next cell (only movers that try to move, one step)
    shared     mover id -> co-location group (movers of one group may share a cell; others not)
    Returns the ids whose move succeeds. Deterministic: priority is ascending mover id.
    """
    trying = {m for m, c in intents.items() if c != positions[m]}
    ok = set(trying)
    changed = True
    while changed:
        changed = False
        # occupancy after this round's moves
        occ: Dict[Cell, List[int]] = {}
        for m, c in positions.items():
            occ.setdefault(intents[m] if m in ok else c, []).append(m)
        for m in sorted(ok):
            tgt = intents[m]
            others = [o for o in occ.get(tgt, []) if o != m]
            clash = [o for o in others if shared.get(o, -1 - o) != shared.get(m, -1 - m)]
            if clash:
                # a stationary / failed occupant blocks; among movers into the same cell, lowest id wins
                if any(o not in ok for o in clash) or min(clash + [m]) != m:
                    ok.discard(m)
                    changed = True
                    break
            # swap: m -> tgt while someone at tgt moves to m's cell
            swap = [o for o in ok if o != m and positions[o] == tgt and intents[o] == positions[m]
                    and shared.get(o, -1 - o) != shared.get(m, -1 - m)]
            if swap:
                ok.discard(m)
                for o in swap:
                    ok.discard(o)
                changed = True
                break
    return ok
