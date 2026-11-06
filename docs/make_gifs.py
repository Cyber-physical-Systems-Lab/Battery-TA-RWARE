"""Render the four README animations (docs/img/*.gif): large versions of basic, intermediate,
advanced and the tutorial scenario, run by the greedy baseline.

    python docs/make_gifs.py            # needs Pillow (pip install pillow)

The map comes from the built-in renderer (``battery_tarware.rendering.rgb_array``, the TA-RWARE
look). This script adds a title bar and a legend:

* shape = robot type;
* fill colour = activity (idle, travelling, waiting for the crew, working, carrying, charging, ...);
* shelf colours: stored, requested (payload waiting), delivery target (purple frame);
* battery badge.
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import battery_tarware as bt  # noqa: E402
from battery_tarware.rendering import legend, regular_polygon, rgb_array, robot_activity  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "img")
CELL = 24
PANEL = 175
HEAD = 28


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                 # Pillow < 10.1
        return ImageFont.load_default()


F_SMALL, F_TEXT, F_HEAD = font(11), font(12), font(14)


def frame(env, title):
    sim = env.unwrapped.sim
    world = Image.fromarray(rgb_array(sim, cell_px=CELL))
    leg = legend(sim)
    rows = 3 + len(leg["types"]) + len(leg["activities"]) + len(leg["cells"])
    h = max(world.height, rows * 16 + 30)
    img = Image.new("RGB", (world.width + PANEL, h + HEAD), (255, 255, 255))
    img.paste(world, (0, HEAD))
    d = ImageDraw.Draw(img)
    m = sim.metrics()
    d.text((4, 7), "%s   tick %d   completed %d   open %d" % (title, sim.tick, m["tasks_completed"],
                                                               m["tasks_open"]), fill="black", font=F_HEAD)
    px, py = world.width + 10, HEAD

    def heading(text):
        nonlocal py
        d.text((px, py), text, fill="black", font=F_TEXT)
        py += 16

    heading("robot type (shape)")
    for name, sides, n in leg["types"]:
        d.polygon(regular_polygon(px + 6, py + 7, 6.5, sides), fill=(255, 255, 255), outline=(0, 0, 0))
        d.text((px + 18, py), "%s x%d" % (name, n), fill=(40, 40, 40), font=F_SMALL)
        py += 15
    py += 5
    heading("activity (colour)")
    for name, col in leg["activities"]:
        d.ellipse([px + 1, py + 2, px + 11, py + 12], fill=col, outline=(0, 0, 0))
        d.text((px + 18, py), name.replace("_", " "), fill=(40, 40, 40), font=F_SMALL)
        py += 15
    py += 5
    heading("cells")
    for name, col in leg["cells"]:
        if name == "delivery target":
            d.rectangle([px, py + 1, px + 12, py + 13], outline=col, width=3)
        else:
            d.rectangle([px, py + 1, px + 12, py + 13], fill=col, outline=(0, 0, 0))
        d.text((px + 18, py), name, fill=(40, 40, 40), font=F_SMALL)
        py += 15
    return img


LARGE_RACKS = {"rack_rows": 4, "rack_cols": 7, "rack_height": 8}   # 4 x 7 blocks of 2 x 8 shelves


def large_map(goals=7, docks=12, first_column=None, bottom_zone=None):
    """The large rack layout as ASCII rows. ``first_column`` = {block row: symbol} turns the shelves
    of the left-most rack block in that block row into a custom zone; ``bottom_zone`` = (symbol, n)
    puts n cells of a custom zone in the middle of the bottom (goal) row."""
    from battery_tarware.layout import generate_ascii
    rows = [list(r) for r in generate_ascii(goals=goals, docks=docks, **LARGE_RACKS).strip("\n").splitlines()]
    blocks, current = [], []
    for y, row in enumerate(rows):
        if "S" in row:
            current.append(y)
        elif current:
            blocks.append(current)
            current = []
    first = rows[blocks[0][0]].index("S")
    for b, sym in (first_column or {}).items():
        for y in blocks[b]:
            rows[y][first] = rows[y][first + 1] = sym
    if bottom_zone:
        sym, n = bottom_zone
        bottom = rows[-1]
        mid = len(bottom) // 2
        free = sorted((i for i, c in enumerate(bottom) if c == "."), key=lambda i: abs(i - mid))
        for i in sorted(free[:n]):
            bottom[i] = sym
    return ["".join(r) for r in rows]


def scale(cfg, k, generator=None, ascii=None, demand=None, initial_soc=None):
    """A large version of a scenario: k times the fleet, the demand (or ``demand`` times) and the
    task board; a bigger generated map, or the ASCII map tiled k times side by side."""
    cfg = copy.deepcopy(cfg)
    cfg["name"] = "%s (large)" % cfg["name"]
    if generator is not None:
        cfg["map"] = {"generator": generator}
    elif ascii is not None:
        cfg["map"] = dict(cfg["map"], ascii=ascii)
    else:
        rows = cfg["map"]["ascii"]
        rows = rows.strip("\n").splitlines() if isinstance(rows, str) else list(rows)
        cfg["map"] = dict(cfg["map"], ascii=[r * k for r in rows])
    for group in cfg["fleet"]:
        group["count"] *= k
        if initial_soc is not None:
            group["initial_soc"] = initial_soc
    f = k if demand is None else demand
    for d in cfg["demand"]:
        if "open" in d:
            d["open"] = int(round(d["open"] * f))
        if "rate" in d:
            d["rate"] *= f
        if "rates" in d:
            d["rates"] = [[t, r * f] for t, r in d["rates"]]
        if "count" in d:
            d["count"] = int(round(d["count"] * f))
    obs = dict(cfg.get("observation", {}))
    obs["task_slots"] = obs.get("task_slots", 16) * k
    cfg["observation"] = obs
    return cfg


def large_scenarios():
    """The four README animations: every scenario scaled up on the same 4 x 7 rack-block layout."""
    import yaml
    tutorial = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "examples", "tutorial.yaml")))
    return {
        # 3x: 18 AGVs, 9 picking robots; part-charged at the start so the shared docks are contested
        "basic": scale(bt.get_scenario("basic"), 3, initial_soc=[0.25, 0.6],
                       generator=dict(LARGE_RACKS, goals=7, docks=8)),
        # 2x: 12 tote AGVs, 4 pallet AGVs, 10 picking robots
        "intermediate": scale(bt.get_scenario("intermediate"), 2, generator=dict(LARGE_RACKS, goals=7, docks=12)),
        # 2x: receiving shelves in the top-left block, quality check in the bottom-left block
        "advanced": scale(bt.get_scenario("advanced"), 2, demand=1.0,
                          ascii=large_map(goals=7, docks=12, first_column={0: "I", 3: "Q"})),
        # 3x: a packing station in the middle of the goal row
        "tutorial": scale(tutorial, 3, demand=2, ascii=large_map(goals=6, docks=10, bottom_zone=("P", 4))),
    }


def _charging_window(scenario, seed, length, **overrides):
    """Start tick of the window with the most robots charging or heading to a charger."""
    env = bt.make(scenario, **overrides)
    coordinator = bt.GreedyCoalitionCoordinator("fifo")
    env.reset(seed=seed)
    coordinator.reset(env)
    counts = []
    while True:
        _, _, terminated, truncated, _ = env.step(coordinator.act(env))
        counts.append(sum(robot_activity(env.sim, r) in ("charging", "to_charger") for r in env.sim.robots))
        if terminated or truncated:
            break
    c = np.concatenate([[0], np.cumsum(counts)])
    sums = c[length:] - c[:-length]
    return int(np.argmax(sums)), int(sums.max())


def record(scenario, frames, every, seed=0, name=None, title=None, **overrides):
    """Record ``frames`` frames (one every ``every`` ticks) over the stretch with the most charging."""
    warmup, charge_ticks = _charging_window(scenario, seed, frames * every, **overrides)
    env = bt.make(scenario, **overrides)
    coordinator = bt.GreedyCoalitionCoordinator("fifo")
    env.reset(seed=seed)
    coordinator.reset(env)
    images = []
    for t in range(warmup + frames * every):
        env.step(coordinator.act(env))
        if t >= warmup and (t - warmup) % every == 0:
            images.append(frame(env, title or name or scenario).convert("P", palette=Image.ADAPTIVE, colors=96))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "%s.gif" % (name or scenario))
    images[0].save(path, save_all=True, append_images=images[1:], duration=120, loop=0, optimize=True)
    print(path, "%.0f kB" % (os.path.getsize(path) / 1024), "ticks %d-%d" % (warmup, warmup + frames * every),
          "charging robot-ticks %d" % charge_ticks)


if __name__ == "__main__":
    for name, cfg in large_scenarios().items():
        record(cfg, frames=160, every=1 if name in ("basic", "tutorial") else 2, name=name, title=cfg["name"])
