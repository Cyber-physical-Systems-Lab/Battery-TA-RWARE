"""Rendering: ``rgb_array`` (numpy image), ``ansi`` (text) and ``human`` (matplotlib window).

The image follows the TA-RWARE look: a white grid with black lines, shelves as padded squares and
robots as regular polygons with a heading tick and a battery badge. Its encodings are:

* **shape = robot type** (hexagon, diamond, triangle, octagon, ... in the order of ``robot_types``);
* **fill colour = what the robot is doing** (``STATUS_COLORS``);
* **shelf colour = what the shelf is for**: dark blue for stored, teal for a payload waiting to be
  picked (a requested shelf), and a purple frame for where a carried payload is going;
* **battery badge**: green above 80 %, yellow above 20 %, red below.

Only numpy is needed. ``docs/make_gifs.py`` adds a title bar and a legend with Pillow.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from battery_tarware.navigation import MOVES
from battery_tarware.robots import RobotStatus

Color = Tuple[int, int, int]

BACKGROUND: Color = (255, 255, 255)
GRID: Color = (0, 0, 0)
WALL: Color = (40, 40, 40)
ZONE_COLORS: Dict[str, Color] = {"goal": (60, 60, 60), "dock": (173, 216, 230)}
CUSTOM_COLORS: List[Color] = [(250, 230, 160), (230, 200, 235), (200, 235, 225), (250, 210, 190)]
SHELF: Color = (72, 61, 139)            # stored
SHELF_REQUESTED: Color = (0, 128, 128)  # payload waiting to be picked / task cell to serve
SHELF_TARGET: Color = (200, 100, 200)   # where a carried payload is going (frame)

STATUS_COLORS: Dict[str, Color] = {
    "idle": (255, 140, 0),          # orange: free
    "travelling": (30, 144, 255),   # blue: on a task, driving to its cell
    "waiting": (150, 90, 200),      # violet: on a task, in place, waiting for the rest of the crew
    "working": (220, 20, 20),       # red: working a stage
    "carrying": (128, 0, 0),        # maroon: driving with a payload
    "to_charger": (120, 200, 120),  # light green: heading to a charger
    "charging": (20, 140, 50),      # green: charging
    "failed": (140, 140, 140),      # grey: broken down, under repair
    "stranded": (0, 0, 0),          # black: battery empty
}
TYPE_SIDES = (6, 4, 3, 8, 5, 7, 10, 12)  # polygon sides per robot type, in config order


def robot_activity(sim, r) -> str:
    """The status category used for the robot's colour."""
    if r.status == RobotStatus.STRANDED:
        return "stranded"
    if r.status == RobotStatus.FAILED:
        return "failed"
    if r.status == RobotStatus.CHARGE:
        return "charging" if r.charging else "to_charger"
    if r.task_id is None:
        return "idle"
    if r.worked:
        return "working"
    if sim.carrying(r):
        return "carrying"
    return "travelling" if r.moved else "waiting"


def type_sides(type_index: int) -> int:
    return TYPE_SIDES[type_index % len(TYPE_SIDES)]


# ---------------------------------------------------------------- numpy drawing primitives
def _rect(img, x0, y0, x1, y1, col):
    img[max(0, int(y0)):max(0, int(y1)), max(0, int(x0)):max(0, int(x1))] = col


def _polygon(img, verts, col):
    """Fill a convex polygon given as [(x, y), ...]."""
    xs, ys = [v[0] for v in verts], [v[1] for v in verts]
    x0, x1 = max(0, int(math.floor(min(xs)))), min(img.shape[1], int(math.ceil(max(xs))) + 1)
    y0, y1 = max(0, int(math.floor(min(ys)))), min(img.shape[0], int(math.ceil(max(ys))) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    px, py = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    inside_pos = np.ones(px.shape, bool)
    inside_neg = np.ones(px.shape, bool)
    for (ax, ay), (bx, by) in zip(verts, verts[1:] + verts[:1]):
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        inside_pos &= cross >= -1e-9
        inside_neg &= cross <= 1e-9
    mask = inside_pos | inside_neg
    img[y0:y1, x0:x1][mask] = col


def regular_polygon(cx, cy, radius, sides) -> List[Tuple[float, float]]:
    """Vertices of a regular polygon, first vertex pointing right (as in TA-RWARE)."""
    if sides >= 10:
        sides = 24                                       # a circle
    return [(cx + radius * math.cos(2 * math.pi * i / sides), cy - radius * math.sin(2 * math.pi * i / sides))
            for i in range(sides)]


def _line(img, p0, p1, col, width=2):
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) * 2 + 1
    h = width / 2
    for t in np.linspace(0.0, 1.0, n):
        x, y = p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])
        _rect(img, round(x - h), round(y - h), round(x + h), round(y + h), col)


def _frame(img, x0, y0, x1, y1, col, w=3):
    _rect(img, x0, y0, x1, y0 + w, col)
    _rect(img, x0, y1 - w, x1, y1, col)
    _rect(img, x0, y0, x0 + w, y1, col)
    _rect(img, x1 - w, y0, x1, y1, col)


# ---------------------------------------------------------------- the frame
def rgb_array(sim, cell_px: int = 30, crew_lines: bool = True) -> np.ndarray:
    """Draw the current state. The image is ``1 + H * (cell_px + 1)`` by ``1 + W * (cell_px + 1)``."""
    H, W = sim.layout.shape
    g = cell_px + 1
    img = np.empty((1 + H * g, 1 + W * g, 3), np.uint8)
    img[:] = GRID
    custom = [z for z in sim.layout.zones if z not in ZONE_COLORS and z not in ("aisle", "storage")]
    pad = max(2, cell_px // 15)

    def box(c):                                            # pixel box of a cell (inside the grid lines)
        return 1 + c[1] * g, 1 + c[0] * g, 1 + c[1] * g + cell_px, 1 + c[0] * g + cell_px

    def center(c):
        x0, y0, x1, y1 = box(c)
        return (x0 + x1) / 2, (y0 + y1) / 2

    # floor and zones
    for y in range(H):
        for x in range(W):
            z = sim.layout.zone_of((y, x))
            if z is None:
                col = WALL
            elif z in ZONE_COLORS:
                col = ZONE_COLORS[z]
            elif z in custom:
                col = CUSTOM_COLORS[custom.index(z) % len(CUSTOM_COLORS)]
            else:
                col = BACKGROUND
            _rect(img, *box((y, x)), col)
            if z == "storage":
                x0, y0, x1, y1 = box((y, x))
                _rect(img, x0 + pad, y0 + pad, x1 - pad, y1 - pad, SHELF)

    # tasks: requested shelves (payload waiting, or the cell a non-carrier task is served at) and targets
    live = [t for t in sim.tasks.values() if not t.done and t.state.value != "pending"]
    for t in live:
        if t.payload_cell is not None:
            x0, y0, x1, y1 = box(t.payload_cell)
            _rect(img, x0 + pad, y0 + pad, x1 - pad, y1 - pad, SHELF_REQUESTED)
        elif not t.carried:
            x0, y0, x1, y1 = box(t.stage_cell)
            _rect(img, x0 + pad, y0 + pad, x1 - pad, y1 - pad, SHELF_REQUESTED)
        if t.carried:
            _frame(img, *box(t.stage_cell), SHELF_TARGET, w=max(2, cell_px // 10))

    # crew lines: committed robots that are not yet in place, to the cell they are heading for
    if crew_lines:
        for t in live:
            for rid in t.robots():
                r = sim.robots[rid]
                if r.cell != t.stage_cell and not sim.carrying(r) and robot_activity(sim, r) == "travelling":
                    _line(img, center(r.cell), center(t.stage_cell), (150, 190, 235), width=1)

    # robots: carried payload underneath, polygon by type, colour by activity, heading, battery
    by_cell: Dict[Tuple[int, int], List] = {}
    for r in sim.robots:
        by_cell.setdefault(r.cell, []).append(r)
    radius = cell_px * 0.4
    for cell, rs in by_cell.items():
        cx0, cy0 = center(cell)
        if any(sim.carrying(r) for r in rs):
            x0, y0, x1, y1 = box(cell)
            _rect(img, x0 + pad, y0 + pad, x1 - pad, y1 - pad, SHELF_REQUESTED)
        n = len(rs)
        rad = radius if n == 1 else radius * 0.6
        for k, r in enumerate(rs):
            if n == 1:
                cx, cy = cx0, cy0
            elif n == 2:                                   # co-located pair (AGV + picker, joint carriers): side by side
                cx, cy = cx0 + cell_px * 0.24 * (2 * k - 1), cy0 - cell_px * 0.08
            else:
                a = 2 * math.pi * k / n + math.pi / 4
                cx, cy = cx0 + cell_px * 0.24 * math.cos(a), cy0 + cell_px * 0.24 * math.sin(a)
            verts = regular_polygon(cx, cy, rad, type_sides(r.type_index))
            _polygon(img, verts, (0, 0, 0))                # outline
            _polygon(img, regular_polygon(cx, cy, rad - 1.2, type_sides(r.type_index)),
                     STATUS_COLORS[robot_activity(sim, r)])
            dy, dx = MOVES[r.heading]
            _line(img, (cx, cy), (cx + dx * rad, cy + dy * rad), (0, 0, 0), width=2)
            # battery badge (bottom right of the cell)
            bx0, by0, bx1, by1 = box(cell)
            bw, bh = max(6, cell_px // 3), max(3, cell_px // 7)
            ox = bx1 - bw - 2 - (k * (bw + 1) if n > 1 else 0)
            oy = by1 - bh - 2
            soc = max(0.0, min(1.0, r.soc))
            col = (0, 200, 0) if soc > 0.8 else (240, 200, 0) if soc > 0.2 else (230, 0, 0)
            _rect(img, ox - 1, oy - 1, ox + bw + 1, oy + bh + 1, (0, 0, 0))
            _rect(img, ox, oy, ox + bw, oy + bh, (255, 255, 255))
            _rect(img, ox, oy, ox + max(1, round(bw * soc)), oy + bh, col)
    return img


def legend(sim) -> Dict[str, list]:
    """What the image encodes, for drawing a legend: robot types (name, sides, count), activities
    (name, colour) and cells (name, colour)."""
    types = [(name, type_sides(i), sum(r.type_index == i for r in sim.robots))
             for i, name in enumerate(sim.type_names)]
    custom = [z for z in sim.layout.zones if z not in ZONE_COLORS and z not in ("aisle", "storage")]
    cells = [("shelf", SHELF), ("requested shelf", SHELF_REQUESTED), ("delivery target", SHELF_TARGET)]
    cells += [(z, ZONE_COLORS[z]) for z in ("goal", "dock")]
    cells += [(z, CUSTOM_COLORS[i % len(CUSTOM_COLORS)]) for i, z in enumerate(custom)]
    return {"types": [t for t in types if t[2]], "activities": list(STATUS_COLORS.items()), "cells": cells}


def ansi(sim) -> str:
    rows = [list(line) for line in sim.layout.to_ascii().splitlines()]
    for r in sim.robots:
        y, x = r.cell
        ch = sim.type_names[r.type_index][0]
        rows[y][x] = ch.upper() if sim.carrying(r) else ch.lower()
    head = "tick %d | open tasks %d | done %d" % (sim.tick, sum(not t.done for t in sim.tasks.values()),
                                                  sum(t.state.value == "completed" for t in sim.tasks.values()))
    return head + "\n" + "\n".join("".join(r) for r in rows)


class HumanViewer:
    def __init__(self):
        import matplotlib.pyplot as plt   # optional dependency
        self.plt = plt
        self.fig, self.ax = plt.subplots()
        self.im = None

    def show(self, img: np.ndarray, title: str) -> None:
        if self.im is None:
            self.im = self.ax.imshow(img)
            self.ax.axis("off")
        else:
            self.im.set_data(img)
        self.ax.set_title(title)
        self.plt.pause(0.001)

    def close(self) -> None:
        self.plt.close(self.fig)
