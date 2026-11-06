"""Trajectory fingerprints of task-only runs, to show that the order layer leaves them unchanged.

    python validation/orders/identity.py before.json     # on the base commit (e.g. 93658ea)
    python validation/orders/identity.py after.json      # on this branch
    python -c "import json; a, b = (json.load(open(f)) for f in ('before.json', 'after.json')); print(a == b)"

Built-ins basic / intermediate / advanced x seeds 0-2 x {greedy, random} for 600 ticks, primitive
mode for 300 ticks, and intermediate with role alternatives: robot cells, energy, status, task and
role every tick, rewards, the full event log and the final metrics.
"""
import hashlib, json, sys
import numpy as np
import battery_tarware as bt

def run(scn, seed, coord, ticks, **kw):
    env = bt.make(scn, max_steps=ticks, **kw)
    sim = env.unwrapped.sim
    env.reset(seed=seed)
    h = hashlib.sha256()
    co = bt.GreedyCoalitionCoordinator("fifo") if coord == "greedy" else bt.RandomCoordinator(seed)
    for _ in range(ticks):
        _, rew, term, trunc, info = env.step(co.act(env))
        h.update(repr([(r.cell, round(r.energy_wh, 6), r.status.value, r.task_id, r.role) for r in sim.robots]).encode())
        h.update(repr(round(float(rew), 6)).encode())
        if term or trunc: break
    h.update(repr([(e.tick, e.type, e.robots, e.task, sorted(e.data.items())) for e in sim.events.events]).encode())
    h.update(json.dumps(sim.metrics(), sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]

out = {}
for scn in ("basic", "intermediate", "advanced"):
    for seed in (0, 1, 2):
        for coord in ("greedy", "random"):
            out["%s/%d/%s" % (scn, seed, coord)] = run(scn, seed, coord, 600)
    out["%s/0/greedy/primitive" % scn] = run(scn, 0, "random", 300, action_mode="primitive")
out["intermediate/0/greedy/subst"] = None
try:
    env_cfg = bt.get_scenario("intermediate", substitution=True)
    out["intermediate/0/greedy/subst"] = run(env_cfg, 0, "greedy", 600)
except Exception as e:
    out["intermediate/0/greedy/subst"] = "ERR %s" % e
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(len(out), "fingerprints")
