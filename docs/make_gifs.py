"""Render the README animations (docs/img/*.gif) with the greedy baseline.

    python docs/make_gifs.py            # needs Pillow (pip install pillow)

The map comes from the built-in renderer (``battery_tarware.rendering.rgb_array``, the TA-RWARE
look). This script adds a title bar and a legend:

* shape = robot type;
* fill colour = activity (idle, travelling, waiting for the crew, working, carrying, charging, ...);
* shelf colours: stored, requested (payload waiting), delivery target (purple frame);
* battery badge.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import battery_tarware as bt  # noqa: E402
from battery_tarware.rendering import legend, regular_polygon, rgb_array  # noqa: E402

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


def large_config():
    """The basic scenario scaled up 3x: an 18-block warehouse, 18 AGVs and 9 pickers."""
    cfg = bt.get_scenario("basic")
    cfg["name"] = "basic_large"
    cfg["map"] = {"generator": {"rack_rows": 3, "rack_cols": 6, "rack_height": 5, "goals": 6, "docks": 8}}
    for group in cfg["fleet"]:
        group["count"] *= 3
    cfg["demand"][0]["open"] *= 3
    cfg["observation"] = {"task_slots": 32}
    return cfg


def record(scenario, frames, every, seed=0, warmup=0, name=None, **overrides):
    env = bt.make(scenario, **overrides)
    coordinator = bt.GreedyCoalitionCoordinator("fifo")
    env.reset(seed=seed)
    coordinator.reset(env)
    images = []
    for t in range(warmup + frames * every):
        env.step(coordinator.act(env))
        if t >= warmup and (t - warmup) % every == 0:
            images.append(frame(env, name or scenario).convert("P", palette=Image.ADAPTIVE, colors=96))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "%s.gif" % (name or scenario))
    images[0].save(path, save_all=True, append_images=images[1:], duration=120, loop=0, optimize=True)
    print(path, "%.0f kB" % (os.path.getsize(path) / 1024))


if __name__ == "__main__":
    record(large_config(), frames=160, every=1, warmup=300, name="large")
    record("basic", frames=180, every=1, warmup=100)
    record("intermediate", frames=180, every=2, warmup=1000)   # demand peak
    record("advanced", frames=180, every=2, warmup=850)
    record(os.path.join(os.path.dirname(__file__), "..", "examples", "tutorial.yaml"), frames=180, every=1,
           warmup=1050, name="tutorial")
