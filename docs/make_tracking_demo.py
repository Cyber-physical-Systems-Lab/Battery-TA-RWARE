"""Package and robot tracking demo: the map with a live status board, plus CSV ledgers.

    python docs/make_tracking_demo.py [scenario] [seed] [ticks]      # needs Pillow

Writes docs/img/tracking_<scenario>.gif (map, package board, robot board) and
docs/img/tracking_<scenario>_{packages,robots}.csv (the full ledgers at the end of the run).
The run itself is the greedy baseline; the tracker only observes it.
"""
from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import battery_tarware as bt  # noqa: E402
from make_gifs import F_SMALL, F_TEXT, frame  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "img")
STATUS_COL = {"waiting": (150, 150, 150), "assigned": (70, 130, 220), "loading": (200, 60, 60),
              "in_transit": (130, 30, 30), "unloading": (200, 60, 60), "returning": (160, 90, 200),
              "dropped": (230, 140, 20), "delivered": (40, 150, 70), "stored": (40, 150, 70),
              "cancelled": (90, 90, 90)}
BOARD_W = 470
ROWS = 22


def board(tracker, height):
    sim = tracker.sim
    img = Image.new("RGB", (BOARD_W, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    y = 6
    counts = bt.status_counts(tracker)
    done = counts.get("delivered", 0) + counts.get("stored", 0)
    d.text((8, y), "packages: %d released, %d done   %s" % (
        len(tracker.packages), done, "  ".join("%s %d" % (k, v) for k, v in sorted(counts.items())
                                                 if k not in ("delivered", "stored"))), fill="black", font=F_TEXT)
    y += 20
    cols = [(8, "package"), (66, "status"), (138, "rack"), (186, "goal"), (234, "at"), (282, "handled by"),
            (420, "dist")]
    for x, h in cols:
        d.text((x, y), h, fill=(60, 60, 60), font=F_SMALL)
    y += 15
    active = [p for p in tracker.packages.values() if p.status not in ("delivered", "stored", "cancelled")]
    recent = sorted((p for p in tracker.packages.values() if p.status in ("delivered", "stored")),
                    key=lambda p: -(p.completed or 0))
    shown = (sorted(active, key=lambda p: p.released) + recent)[:ROWS]
    for p in shown:
        crew = sorted({(h.robot, h.role) for h in p.handlers if h.left is None}) or \
            sorted({(h.robot, h.role) for h in p.handlers})[-2:]
        who = ", ".join("%s(%s)" % (sim.robots[r].name.replace("picking_robot", "pick").replace("_", ""), role[:5])
                        for r, role in crew)
        vals = [p.package_id, p.status, "%d,%d" % p.rack, "%d,%d" % p.goal,
                "%d,%d" % p.location if p.location else "-", who[:26], str(p.distance_loaded)]
        d.rectangle([62, y + 2, 64, y + 11], fill=STATUS_COL.get(p.status, (0, 0, 0)))
        for (x, _), v in zip(cols, vals):
            d.text((x, y), v, fill=STATUS_COL[p.status] if x == 66 else (30, 30, 30), font=F_SMALL)
        y += 14
    y += 8
    d.text((8, y), "robots", fill="black", font=F_TEXT)
    y += 18
    rcols = [(8, "robot"), (120, "activity"), (196, "battery"), (250, "package"), (310, "dist e/l")]
    for x, h in rcols:
        d.text((x, y), h, fill=(60, 60, 60), font=F_SMALL)
    y += 15
    for r in tracker.robots.values():
        if y > height - 14:
            break
        vals = [r.name, r.activity, "%3.0f%%" % (100 * r.soc), r.package or "-",
                "%d / %d" % (r.distance_empty, r.distance_loaded)]
        for (x, _), v in zip(rcols, vals):
            d.text((x, y), v, fill=(200, 40, 40) if x == 196 and r.soc < 0.3 else (30, 30, 30), font=F_SMALL)
        y += 13
    return img


def main(scenario="basic", seed=0, ticks=400, every=2):
    env = bt.make(scenario, max_steps=ticks)
    sim = env.unwrapped.sim
    co = bt.GreedyCoalitionCoordinator("fifo")
    env.reset(seed=seed)
    tracker = bt.PackageTracker(sim)
    images = []
    for t in range(ticks):
        env.step(co.act(env))
        tracker.update()
        if t % every == 0:
            left = frame(env, "%s - package tracking" % scenario)
            right = board(tracker, max(left.height, 640))
            img = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), (255, 255, 255))
            img.paste(left, (0, 0))
            img.paste(right, (left.width, 0))
            images.append(img.convert("P", palette=Image.ADAPTIVE, colors=96))
    os.makedirs(OUT, exist_ok=True)
    gif = os.path.join(OUT, "tracking_%s.gif" % scenario)
    images[0].save(gif, save_all=True, append_images=images[1:], duration=150, loop=0, optimize=True)
    bt.write_csv(bt.package_table(tracker), os.path.join(OUT, "tracking_%s_packages.csv" % scenario))
    bt.write_csv(bt.robot_table(tracker), os.path.join(OUT, "tracking_%s_robots.csv" % scenario))
    print(gif, "%.0f kB" % (os.path.getsize(gif) / 1024), bt.status_counts(tracker))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "basic", int(a[1]) if len(a) > 1 else 0, int(a[2]) if len(a) > 2 else 400)
