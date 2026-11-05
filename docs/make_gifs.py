"""Render the README animations (docs/img/*.gif) with the greedy baseline.

    python docs/make_gifs.py            # needs Pillow (pip install pillow)

Each frame shows the map zones, robots coloured by type (letter = type initial, bar = battery,
yellow cap = carrying), open tasks (red marker at the payload / current stage cell) and crew lines
from every assigned robot to its task's current stage cell.
"""
from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import battery_tarware as bt  # noqa: E402
from battery_tarware.robots import RobotStatus  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "img")
ZONE = {"aisle": (238, 238, 238), "storage": (196, 168, 128), "goal": (122, 196, 132),
        "dock": (128, 166, 222)}
CUSTOM = [(226, 206, 128), (206, 150, 206), (146, 206, 206), (236, 176, 146)]
WALL = (70, 70, 70)
TYPE = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14),
        (140, 86, 75), (227, 119, 194), (23, 190, 207)]
CELL = 22
PANEL = 190


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                 # Pillow < 10.1
        return ImageFont.load_default()


F_SMALL, F_TEXT, F_HEAD = font(11), font(13), font(15)


def frame(env, title):
    sim = env.unwrapped.sim
    H, W = sim.layout.shape
    img = Image.new("RGB", (W * CELL + PANEL, H * CELL + 30), (255, 255, 255))
    d = ImageDraw.Draw(img)
    custom = [z for z in sim.layout.zones if z not in ZONE]
    oy = 30
    for y in range(H):
        for x in range(W):
            z = sim.layout.zone_of((y, x))
            col = WALL if z is None else ZONE.get(z) or CUSTOM[custom.index(z) % len(CUSTOM)]
            d.rectangle([x * CELL, oy + y * CELL, (x + 1) * CELL - 1, oy + (y + 1) * CELL - 1], fill=col,
                        outline=(215, 215, 215))
    center = lambda c: (c[1] * CELL + CELL / 2, oy + c[0] * CELL + CELL / 2)
    for t in sim.tasks.values():
        if t.done or t.state.value == "pending":
            continue
        cx, cy = center(t.payload_cell or t.stage_cell)
        d.polygon([(cx, cy - 6), (cx + 6, cy), (cx, cy + 6), (cx - 6, cy)], fill=(225, 50, 50))
        for rid in t.robots():
            r = sim.robots[rid]
            d.line([center(r.cell), center(t.stage_cell)], fill=(225, 50, 50), width=1)
    for r in sim.robots:
        cx, cy = center(r.cell)
        col = TYPE[r.type_index % len(TYPE)]
        if r.status in (RobotStatus.FAILED, RobotStatus.STRANDED):
            col = (120, 120, 120)
        d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=col, outline=(30, 30, 30))
        if sim.carrying(r):
            d.chord([cx - 8, cy - 8, cx + 8, cy + 8], 200, 340, fill=(250, 215, 40))
        d.text((cx, cy + 1), sim.type_names[r.type_index][0].upper(), fill="white", font=F_SMALL, anchor="mm")
        bar = int(round((CELL - 4) * r.soc))
        x0, y1 = r.cell[1] * CELL + 2, oy + (r.cell[0] + 1) * CELL - 3
        d.rectangle([x0, y1, x0 + CELL - 4, y1 + 1], fill=(200, 200, 200))
        d.rectangle([x0, y1, x0 + bar, y1 + 1], fill=(0, 170, 0) if r.soc > 0.25 else (220, 0, 0))
    m = sim.metrics()
    d.text((6, 7), "%s   tick %d   completed %d   open %d" % (title, sim.tick, m["tasks_completed"],
                                                               m["tasks_open"]), fill="black", font=F_HEAD)
    px, py = W * CELL + 12, oy + 4
    d.text((px, py), "robot types", fill="black", font=F_TEXT)
    py += 20
    for i, name in enumerate(sim.type_names):
        n = sum(r.type_index == i for r in sim.robots)
        if not n:
            continue
        d.ellipse([px, py, px + 12, py + 12], fill=TYPE[i % len(TYPE)])
        d.text((px + 18, py), "%s x%d" % (name, n), fill="black", font=F_TEXT)
        py += 18
    py += 10
    d.text((px, py), "zones", fill="black", font=F_TEXT)
    py += 20
    for z in sim.layout.zones:
        col = ZONE.get(z) or CUSTOM[custom.index(z) % len(CUSTOM)]
        d.rectangle([px, py, px + 12, py + 12], fill=col, outline=(150, 150, 150))
        d.text((px + 18, py), z, fill="black", font=F_TEXT)
        py += 18
    py += 10
    for text in ("red: open task + crew", "yellow cap: carrying", "bar: battery"):
        d.text((px, py), text, fill=(80, 80, 80), font=F_SMALL)
        py += 15
    return img


def record(scenario, frames, every, seed=0, warmup=0, name=None, **overrides):
    env = bt.make(scenario, **overrides)
    coordinator = bt.GreedyCoalitionCoordinator("fifo")
    env.reset(seed=seed)
    coordinator.reset(env)
    images = []
    for t in range(warmup + frames * every):
        env.step(coordinator.act(env))
        if t >= warmup and (t - warmup) % every == 0:
            images.append(frame(env, name or scenario).convert("P", palette=Image.ADAPTIVE, colors=64))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "%s.gif" % (name or scenario))
    images[0].save(path, save_all=True, append_images=images[1:], duration=120, loop=0, optimize=True)
    print(path, "%.0f kB" % (os.path.getsize(path) / 1024))


if __name__ == "__main__":
    record("basic", frames=180, every=1, warmup=100)
    record("intermediate", frames=180, every=2, warmup=1000)   # demand peak
    record("advanced", frames=180, every=2, warmup=850)
    record(os.path.join(os.path.dirname(__file__), "..", "examples", "tutorial.yaml"), frames=180, every=1,
           warmup=1050, name="tutorial")
