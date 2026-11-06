"""Grid maps: zones per cell, from ASCII art or a parametric warehouse generator.

Coordinates are ``(row, col)`` = ``(y, x)`` everywhere in this package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from battery_tarware.config import CELL_ZONES, ConfigError, MapSpec

Cell = Tuple[int, int]
WALL = -1
BUILTIN_SYMBOLS = {".": "aisle", "+": "aisle", "S": "storage", "G": "goal", "C": "dock",
                   "^": "aisle", "v": "aisle", "<": "aisle", ">": "aisle"}
# one-way aisle cells: the direction a robot may travel along the street (MOVES order: up, down,
# left, right). On a one-way cell a robot may move on in that direction or turn off sideways into a
# non-street cell (storage, goal, dock, plain aisle); it may never enter a one-way cell against its
# direction. '+' marks a crossing: any direction, still a street cell.
DIRECTION_SYMBOLS = {"^": 0, "v": 1, "<": 2, ">": 3, "+": -2}
FREE, CROSSING = -1, -2
OPPOSITE = (1, 0, 3, 2)


def generate_ascii(rack_rows: int = 2, rack_cols: int = 3, rack_height: int = 4, goals: int = 2,
                   docks: int = 4, aisle_width: int = 2, one_way: bool = False) -> str:
    """RWARE-style warehouse: blocks of 2-wide storage columns separated by ``aisle_width``-wide
    aisles, a dock row at the top and a bottom row of goal (workstation) cells joined by aisle, so no
    goal is a dead end. With ``aisle_width >= 2`` a robot waiting in an aisle never closes it.
    ``one_way=True`` (only with ``aisle_width == 1``) turns the aisles into a one-way street grid."""
    if min(rack_rows, rack_cols, rack_height, aisle_width) < 1 or goals < 1 or docks < 0:
        raise ConfigError("generator: rack_rows, rack_cols, rack_height, aisle_width, goals must be >= 1, docks >= 0")
    if one_way and aisle_width != 1:
        raise ConfigError("generator: one_way needs aisle_width == 1")
    a = aisle_width
    period = 2 + a
    W = rack_cols * period + a
    if goals > W or docks > W:
        raise ConfigError("generator: at most %d goals / docks fit a %d-wide map" % (W, W))
    rows: List[str] = []

    def spread(n: int) -> List[int]:
        return sorted({int(round((i + 0.5) * W / n - 0.5)) for i in range(n)}) if n else []

    dock_cols = set(spread(docks))
    rows.append("".join("C" if x in dock_cols else "." for x in range(W)))
    rows.extend(["." * W] * a)
    for _ in range(rack_rows):
        line = "".join("." if x % period < a else "S" for x in range(W))
        rows.extend([line] * rack_height)
        rows.extend(["." * W] * a)
    goal_cols = set(spread(goals))
    rows.append("".join("G" if x in goal_cols else "." for x in range(W)))
    if one_way:
        rows = _orient(rows, rack_cols)
    return "\n".join(rows)


def dock_bays(text: str) -> str:
    """Off-corridor docks: every row made only of dock ('C') and plain aisle ('.') cells, with at
    least one dock, has its aisle cells turned into walls. Each dock then becomes a one-cell bay that
    opens onto the row(s) next to it, so a robot charging there blocks no through traffic."""
    rows = [ln.strip() for ln in text.strip("\n").splitlines() if ln.strip()]
    out = []
    for y, ln in enumerate(rows):
        if "C" in ln and set(ln) <= {"C", "."}:
            ln = ln.replace(".", "#")
            for x, ch in enumerate(ln):
                if ch != "C":
                    continue
                nbrs = [rows[y + dy][x] for dy in (-1, 1) if 0 <= y + dy < len(rows)]
                if not any(c not in "#C" for c in nbrs):
                    raise ConfigError("map: dock at (%d, %d) has no approach once its row becomes bays" % (y, x))
        out.append(ln)
    return "\n".join(out)


def _orient(rows: List[str], rack_cols: int) -> List[str]:
    """One-way street grid: corridor columns alternate down/up and street rows alternate, with the
    outer streets forming a loop (down the left, right along the bottom, up the right, left along
    the top), so every cell stays reachable. Street crossings ('+') allow any direction."""
    H, W = len(rows), len(rows[0])
    grid = [list(r) for r in rows]
    cols = list(range(0, W, 3))
    street_rows = [y for y in range(1, H - 1) if all(ch == "." for ch in rows[y])]
    col_dir = {x: ("v" if i % 2 == 0 else "^") for i, x in enumerate(cols)}
    col_dir[cols[0]], col_dir[cols[-1]] = "v", "^"
    row_dir = {y: ("<" if i % 2 == 0 else ">") for i, y in enumerate(street_rows)}
    row_dir[street_rows[0]], row_dir[street_rows[-1]] = "<", ">"
    # the dock row and the goal row become return lanes next to the outer streets
    row_dir[0], row_dir[H - 1] = ">", "<"
    for y in range(0, H):
        for x in range(W):
            if grid[y][x] != ".":
                continue
            if y in row_dir:
                grid[y][x] = "+" if x in col_dir else row_dir[y]
            elif x in col_dir:
                grid[y][x] = col_dir[x]
    return ["".join(r) for r in grid]


@dataclass
class Layout:
    zones: Tuple[str, ...]              # zone index -> name; WALL (-1) is not traversable
    grid: np.ndarray                    # (H, W) int8 zone index or WALL
    directions: Optional[np.ndarray] = None   # (H, W) int8: one-way exit direction, -1 = any
    symbols: Optional[List[str]] = None       # the ASCII rows the layout was built from

    @staticmethod
    def from_spec(spec: MapSpec, zone_names: Sequence[str]) -> "Layout":
        text = spec.ascii if spec.ascii is not None else generate_ascii(**dict(spec.generator))
        if getattr(spec, "dock_layout", "inline") == "bays":
            text = dock_bays(text)
        lay = Layout.from_ascii(text, spec.zones, zone_names)
        if getattr(spec, "dock_layout", "inline") == "bays" and lay.one_way:
            raise ConfigError("map: dock_layout 'bays' cannot be combined with one-way streets")
        return lay

    @staticmethod
    def from_ascii(text: str, custom: Mapping[str, str], zone_names: Sequence[str]) -> "Layout":
        lines = [ln.strip() for ln in text.strip("\n").splitlines() if ln.strip()]
        if not lines:
            raise ConfigError("map: empty")
        W = len(lines[0])
        if any(len(ln) != W for ln in lines):
            raise ConfigError("map: all rows must have the same width")
        symbols = {**BUILTIN_SYMBOLS, **dict(custom)}
        index = {z: i for i, z in enumerate(zone_names)}
        grid = np.full((len(lines), W), WALL, dtype=np.int8)
        dirs = np.full((len(lines), W), -1, dtype=np.int8)
        for y, ln in enumerate(lines):
            for x, ch in enumerate(ln):
                if ch == "#":
                    continue
                if ch not in symbols:
                    raise ConfigError("map: unknown symbol %r at (%d, %d)" % (ch, y, x))
                grid[y, x] = index[symbols[ch]]
                dirs[y, x] = DIRECTION_SYMBOLS.get(ch, -1)
        lay = Layout(tuple(zone_names), grid, dirs, lines)
        if not lay.cells("aisle"):
            raise ConfigError("map: needs aisle cells")
        if not lay.cells("goal") and not any(z not in CELL_ZONES for z in zone_names):
            raise ConfigError("map: needs at least one goal cell")
        return lay

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.grid.shape)

    def zone_of(self, cell: Cell) -> Optional[str]:
        z = int(self.grid[cell])
        return None if z == WALL else self.zones[z]

    def cells(self, zone: str) -> List[Cell]:
        if zone not in self.zones:
            return []
        ys, xs = np.nonzero(self.grid == self.zones.index(zone))
        return sorted(zip(ys.tolist(), xs.tolist()))

    def passable_mask(self, zones: Sequence[str]) -> np.ndarray:
        idx = [self.zones.index(z) for z in zones if z in self.zones]
        return np.isin(self.grid, idx)

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.grid.shape[0] and 0 <= cell[1] < self.grid.shape[1]

    @property
    def one_way(self) -> bool:
        return self.directions is not None and bool((self.directions != FREE).any())

    def move_ok(self, a: Cell, b: Cell, k: int) -> bool:
        """May a robot step from ``a`` to its neighbour ``b`` (move index ``k``)?"""
        if self.directions is None:
            return True
        da, db = int(self.directions[a]), int(self.directions[b])
        if da >= 0 and da != k and db != FREE:
            return False                         # on a one-way street: go on, or turn off into a side cell
        return not (db >= 0 and db == OPPOSITE[k])

    def to_ascii(self) -> str:
        if self.symbols is not None:
            return "\n".join(self.symbols)
        sym = {"aisle": ".", "storage": "S", "goal": "G", "dock": "C"}
        out = []
        for row in self.grid:
            out.append("".join("#" if z == WALL else sym.get(self.zones[z], self.zones[z][0].lower()) for z in row))
        return "\n".join(out)
