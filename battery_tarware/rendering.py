"""Rendering: ``rgb_array`` (numpy image), ``ansi`` (text) and ``human`` (matplotlib window)."""
from __future__ import annotations

import numpy as np

from battery_tarware.robots import RobotStatus

ZONE_COLORS = {"aisle": (235, 235, 235), "storage": (180, 150, 110), "goal": (110, 190, 120),
               "dock": (120, 160, 220)}
CUSTOM_COLORS = [(220, 200, 120), (200, 140, 200), (140, 200, 200), (230, 170, 140)]
ROBOT_COLORS = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14),
                (140, 86, 75), (227, 119, 194), (23, 190, 207)]
WALL = (60, 60, 60)


def rgb_array(sim, cell_px: int = 16) -> np.ndarray:
    H, W = sim.layout.shape
    img = np.zeros((H * cell_px, W * cell_px, 3), np.uint8)
    custom = [z for z in sim.layout.zones if z not in ZONE_COLORS]
    for y in range(H):
        for x in range(W):
            z = sim.layout.zone_of((y, x))
            col = WALL if z is None else ZONE_COLORS.get(z) or CUSTOM_COLORS[custom.index(z) % len(CUSTOM_COLORS)]
            img[y * cell_px:(y + 1) * cell_px, x * cell_px:(x + 1) * cell_px] = col
    img[::cell_px, :] = img[::cell_px, :] * 0.85
    img[:, ::cell_px] = img[:, ::cell_px] * 0.85
    for t in sim.tasks.values():                 # open task cells
        if t.done or t.state.value == "pending":
            continue
        cy, cx = t.payload_cell or t.stage_cell
        img[cy * cell_px + 2:cy * cell_px + 5, cx * cell_px + 2:(cx + 1) * cell_px - 2] = (240, 60, 60)
    q = cell_px // 4
    for r in sim.robots:
        y, x = r.cell
        col = ROBOT_COLORS[r.type_index % len(ROBOT_COLORS)]
        if r.status in (RobotStatus.FAILED, RobotStatus.STRANDED):
            col = (90, 90, 90)
        img[y * cell_px + q:(y + 1) * cell_px - q, x * cell_px + q:(x + 1) * cell_px - q] = col
        if sim.carrying(r):
            img[y * cell_px + q:y * cell_px + 2 * q, x * cell_px + q:(x + 1) * cell_px - q] = (250, 220, 40)
        bar = int(round((cell_px - 2) * r.soc))
        img[(y + 1) * cell_px - 2:(y + 1) * cell_px - 1, x * cell_px + 1:x * cell_px + 1 + bar] = (0, 200, 0) if r.soc > 0.2 else (230, 0, 0)
    return img


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
