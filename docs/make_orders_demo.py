"""Order-layer demo GIF: the warehouse map beside an order-task kanban (one column per state).

    python docs/make_orders_demo.py [rule] [seed] [ticks]        # needs Pillow

Orders: Poisson outbound / inbound / relocation orders, plus every 40 ticks a two-task order
"relocate the shelf, then ship its package" (the outbound task is *blocked* until the relocation
completes). Writes docs/img/orders_kanban.gif (compact: small frames, every 5th tick, 48 colours).
"""
from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import battery_tarware as bt  # noqa: E402
from battery_tarware.orders.scenarios import order_scenario  # noqa: E402
from battery_tarware.orders.scheduler import OrderScheduler  # noqa: E402
from make_gifs import F_SMALL, F_TEXT, frame  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "img")
COLS = ["blocked", "ready", "assigned", "executing", "suspended", "retrying", "completed"]
COL = {"blocked": (245, 158, 11), "ready": (59, 130, 246), "assigned": (99, 102, 241), "executing": (22, 163, 74),
       "suspended": (168, 85, 247), "retrying": (239, 68, 68), "completed": (5, 150, 105)}
OPS = {"outbound": "OUT", "inbound": "IN", "relocation": "MOVE"}


def board(sim, height, width=560):
    m = sim.orders
    img = Image.new("RGB", (width, height), (246, 247, 249))
    d = ImageDraw.Draw(img)
    s = m.summary()["orders"]
    d.text((8, 6), "orders: %d open, %d completed, %d failed    tick %d" % (
        s.get("open", 0), s.get("completed", 0), s.get("failed", 0), sim.tick), fill="black", font=F_TEXT)
    cw = (width - 8) // len(COLS)
    by = {c: [] for c in COLS}
    for t in m.tasks.values():
        st = t.state.value if t.state.value != "pending" else "blocked"
        if st in by:
            by[st].append(t)
    by["completed"] = sorted(by["completed"], key=lambda t: -(t.completed_at or 0))
    for k, c in enumerate(COLS):
        x = 4 + k * cw
        d.rectangle([x, 28, x + cw - 4, height - 4], fill=(236, 239, 243))
        d.text((x + 4, 31), "%s %d" % (c, len(by[c])), fill=COL[c], font=F_SMALL)
        y = 48
        for t in by[c][: (height - 60) // 30]:
            d.rectangle([x + 3, y, x + cw - 7, y + 26], fill="white", outline=(220, 223, 228))
            d.rectangle([x + 3, y, x + 6, y + 26], fill=COL[c])
            d.text((x + 9, y + 1), "%s %s" % (t.task_id[-3:], OPS[t.op]), fill=(20, 20, 20), font=F_SMALL)
            sub = t.package_id or t.shelf_id or ""
            if c == "blocked" and t.reason:
                sub = t.reason.split(":")[0][:9]
            elif t.attempt > 1:
                sub += " try%d" % t.attempt
            d.text((x + 9, y + 13), sub, fill=(110, 110, 110), font=F_SMALL)
            y += 30
    return img


def main(rule="plb", seed=0, ticks=420, every=5):
    src = {"kind": "poisson", "rate": 0.07, "max_orders": 10 ** 6,
           "mix": {"outbound": 0.6, "inbound": 0.25, "relocation": 0.15}}
    env = bt.make(order_scenario(source=src), max_steps=ticks)
    sim = env.unwrapped.sim
    co = OrderScheduler(rule)
    env.reset(seed=seed)
    images = []
    for t in range(ticks):
        if t % 40 == 5:                                   # relocate-then-ship order (dependency)
            inv = sim.orders.inventory
            p = next((p for p in inv.stored_packages() if inv.shelves[p.shelf_id].busy_task is None), None)
            if p is not None:
                sim.orders.submit({"kind": "outbound", "tasks": [
                    {"key": "mv", "op": "relocation", "shelf": p.shelf_id},
                    {"key": "ship", "op": "outbound", "package": p.package_id, "depends_on": ["mv"]}]})
        env.step(co.act(env))
        if t % every == 0:
            left = frame(env, "orders - %s" % rule.upper())
            left = left.resize((left.width * 3 // 4, left.height * 3 // 4))
            right = board(sim, left.height)
            img = Image.new("RGB", (left.width + right.width, left.height), (255, 255, 255))
            img.paste(left, (0, 0))
            img.paste(right, (left.width, 0))
            images.append(img.convert("P", palette=Image.ADAPTIVE, colors=48))
    os.makedirs(OUT, exist_ok=True)
    gif = os.path.join(OUT, "orders_kanban.gif")
    images[0].save(gif, save_all=True, append_images=images[1:], duration=160, loop=0, optimize=True)
    print(gif, "%.0f kB" % (os.path.getsize(gif) / 1024), len(images), "frames", sim.orders.summary())


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "plb", int(a[1]) if len(a) > 1 else 0, int(a[2]) if len(a) > 2 else 420)
