"""Render the annotated explanation figure (docs/img/explanation.png), the map rotated 90° (landscape).

    python docs/make_figure.py          # needs Pillow

Runs the basic scenario with the greedy baseline, picks the tick at which most kinds of element are
visible at once (robots in each activity, requested shelves, a delivery target, a charging robot)
and labels one example of each.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import battery_tarware as bt  # noqa: E402
from battery_tarware.rendering import robot_activity, rgb_array  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "img", "explanation.png")
CELL = 36
MARGIN = 300
TOP = 56


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def elements(sim):
    """One example cell per kind of element -> {label: cell}."""
    found = {}
    name = {i: n for i, n in enumerate(sim.type_names)}
    acts = {"carrying": "{t} carrying a shelf to a goal",
            "working": "{t} working a stage (red)",
            "waiting": "{t} in place, waiting for its crew",
            "travelling": "{t} driving to its task",
            "idle": "{t} idle, free to assign",
            "charging": "{t} charging at a dock"}
    for r in sim.robots:
        a = robot_activity(sim, r)
        if a in acts and a not in found:
            found[a] = (acts[a].format(t=name[r.type_index].upper() if len(name[r.type_index]) <= 3
                                       else name[r.type_index]), r.cell)
    occupied = {r.cell for r in sim.robots}
    occupied |= {t.stage_cell for t in sim.tasks.values() if t.carried and not t.done}   # keep framed goals apart
    out = {label: cell for label, cell in found.values()}
    for t in sim.tasks.values():
        if t.done or t.state.value == "pending":
            continue
        if t.payload_cell is not None and t.payload_cell not in occupied and "requested shelf" not in out:
            out["requested shelf (an order waits here)"] = t.payload_cell
        if t.carried and "target" not in out:
            out["goal; purple frame = target of a carried shelf"] = t.stage_cell
    layout = sim.layout
    H, W = layout.shape
    for zone, label in (("storage", "stored shelf"), ("goal", "goal: delivery point"), ("dock", "charger dock")):
        for y in range(H):
            hit = next(((y, x) for x in range(W) if layout.zone_of((y, x)) == zone and (y, x) not in occupied
                        and (y, x) not in out.values()), None)
            if hit:
                out[label] = hit
                break
    return out


def best_tick(cfg, seed=0, ticks=2000, start=150):
    env = bt.make(cfg)
    c = bt.GreedyCoalitionCoordinator("fifo")
    env.reset(seed=seed)
    c.reset(env)
    best, best_t = -1, 0
    for t in range(1, ticks + 1):
        env.step(c.act(env))
        acts = {robot_activity(env.sim, r) for r in env.sim.robots}
        score = len(acts & {"carrying", "working", "waiting", "travelling", "idle", "charging"})
        score += 3 * ("charging" in acts) + 2 * any(tk.carried for tk in env.sim.tasks.values())
        if t >= start and score > best:
            best, best_t = score, t
    return best_t


def render(cfg, tick, seed=0):
    env = bt.make(cfg)
    c = bt.GreedyCoalitionCoordinator("fifo")
    env.reset(seed=seed)
    c.reset(env)
    for _ in range(tick):
        env.step(c.act(env))
    sim = env.sim
    arr = rgb_array(sim, cell_px=CELL, crew_lines=True)
    WP = arr.shape[1]
    world = Image.fromarray(np.ascontiguousarray(np.rot90(arr)))   # rotated 90 degrees: landscape
    img = Image.new("RGB", (world.width + 2 * MARGIN, world.height + TOP + 50), (255, 255, 255))
    img.paste(world, (MARGIN, TOP))
    d = ImageDraw.Draw(img)
    f_title, f_lab, f_note = font(20), font(14), font(13)
    d.text((img.width / 2, 18), "Battery-TA-RWARE: one tick of the basic scenario (map rotated 90°)", fill="black",
           font=f_title, anchor="mm")
    d.text((img.width / 2, 42), "shape = robot type (hexagon AGV, diamond picking robot)   colour = activity   "
           "badge = battery   tick = heading   blue line = crew member heading to its task", fill=(70, 70, 70), font=f_note,
           anchor="mm")

    def px(cell):
        g = CELL + 1
        if cell[0] == "badge":                                 # the battery badge, bottom right of the cell
            c = cell[1]
            ox, oy = 1 + c[1] * g + CELL - 8, 1 + c[0] * g + CELL - 5
        else:
            ox, oy = 1 + cell[1] * g + CELL / 2, 1 + cell[0] * g + CELL / 2
        return MARGIN + oy, TOP + (WP - 1 - ox)                    # same rotation as the image

    items = elements(sim).items()
    mid = MARGIN + world.width / 2
    sides = {"L": [], "R": []}
    for label, cell in items:
        sides["L" if px(cell)[0] < mid else "R"].append((label, cell))
    while abs(len(sides["L"]) - len(sides["R"])) > 1:        # balance: move the item nearest the middle
        a, b = ("L", "R") if len(sides["L"]) > len(sides["R"]) else ("R", "L")
        move = min(sides[a], key=lambda it: abs(px(it[1])[0] - mid))
        sides[a].remove(move)
        sides[b].append(move)
    for side, its in sides.items():
        its.sort(key=lambda it: px(it[1])[1])
        gap, top, bottom = 26, TOP + 12, TOP + world.height - 12
        ys = []
        for label, cell in its:                               # labels as close to their target row as possible
            ys.append(max(px(cell)[1], ys[-1] + gap if ys else top))
        over = ys[-1] - bottom if ys else 0
        for k in range(len(ys) - 1, -1, -1):                  # push back up if they ran off the bottom
            if over <= 0:
                break
            ys[k] -= over
            over = ys[k - 1] + gap - ys[k] if k else 0
        for (label, cell), ly in zip(its, ys):
            tx, ty = px(cell)
            if side == "L":
                lx = MARGIN - 14
                d.text((lx - 4, ly), label, fill="black", font=f_lab, anchor="rm")
            else:
                lx = MARGIN + world.width + 14
                d.text((lx + 4, ly), label, fill="black", font=f_lab, anchor="lm")
            d.line([(lx, ly), (tx, ty)], fill=(220, 40, 40), width=2)
            d.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], outline=(220, 40, 40), width=2)
    m = sim.metrics()
    d.text((img.width / 2, img.height - 22),
           "tick %d   tasks completed %d   open %d   (greedy baseline, seed %d)"
           % (sim.tick, m["tasks_completed"], m["tasks_open"], seed), fill=(70, 70, 70), font=f_note,
           anchor="mm")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT, optimize=True)
    print(OUT, img.size, "tick", tick)


if __name__ == "__main__":
    cfg = bt.get_scenario("basic")
    render(cfg, best_tick(cfg))
