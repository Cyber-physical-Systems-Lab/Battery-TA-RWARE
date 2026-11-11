"""Trajectory fingerprints: model 2.0/2.1 behaviour must not change when the code changes.

    python experiments/checks/identity.py                 # compare with the committed reference
    python experiments/checks/identity.py --write out.json  # record fingerprints (e.g. on an old commit)

Each fingerprint hashes, every tick, the robots' cells, energy, status, task and role and the reward,
then the full event log and the final metrics. Runs: built-ins basic / intermediate / advanced x
seeds 0-2 x {greedy FIFO, random} for 600 ticks (model 2.1), primitive mode for 300 ticks,
intermediate with role alternatives, the same three scenarios under model 2.0, and the order layer
(base order scenario, rules FIFO / NSP / PLB, seeds 0-1, 600 ticks, model 2.1).

``reference_identity.json`` was recorded on commit 97f3b5e, before model 2.2 existed. Exit status 1
if any fingerprint differs.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "orders"))

import battery_tarware as bt  # noqa: E402


def _hash_run(env, co, ticks):
    sim = env.unwrapped.sim
    h = hashlib.sha256()
    for _ in range(ticks):
        _, rew, term, trunc, _ = env.step(co.act(env))
        h.update(repr([(r.cell, round(r.energy_wh, 6), r.status.value, r.task_id, r.role) for r in sim.robots]).encode())
        h.update(repr(round(float(rew), 6)).encode())
        if term or trunc:
            break
    h.update(repr([(e.tick, e.type, e.robots, e.task, sorted(e.data.items(), key=lambda kv: kv[0]))
                   for e in sim.events.events]).encode())
    h.update(json.dumps(sim.metrics(), sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def run(scn, seed, coord, ticks, **kw):
    env = bt.make(scn, max_steps=ticks, **kw)
    env.reset(seed=seed)
    co = bt.GreedyCoalitionCoordinator("fifo") if coord == "greedy" else bt.RandomCoordinator(seed)
    co.reset(env)
    return _hash_run(env, co, ticks)


def run_orders(rule, seed, ticks):
    import factors
    from battery_tarware.orders.scheduler import OrderScheduler
    cfg = factors.build()
    env = bt.make(cfg, max_steps=ticks)
    env.reset(seed=seed)
    co = OrderScheduler(rule)
    co.reset(env)
    return _hash_run(env, co, ticks)


def fingerprints():
    out = {}
    for scn in ("basic", "intermediate", "advanced"):
        for seed in (0, 1, 2):
            for coord in ("greedy", "random"):
                out["%s/%d/%s" % (scn, seed, coord)] = run(scn, seed, coord, 600)
        out["%s/0/random/primitive" % scn] = run(scn, 0, "random", 300, action_mode="primitive")
        out["%s/0/greedy/model2.0" % scn] = run(bt.get_scenario(scn, model="2.0"), 0, "greedy", 600)
    out["intermediate/0/greedy/subst"] = run(bt.get_scenario("intermediate", substitution=True), 0, "greedy", 600)
    for rule in ("fifo", "nsp", "plb"):
        for seed in (0, 1):
            out["orders_base/%d/%s" % (seed, rule)] = run_orders(rule, seed, 600)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", help="write fingerprints to this file instead of comparing")
    ap.add_argument("--reference", default=os.path.join(HERE, "reference_identity.json"))
    a = ap.parse_args(argv)
    fp = fingerprints()
    if a.write:
        with open(a.write, "w") as f:
            json.dump(fp, f, indent=1, sort_keys=True)
        print("%d fingerprints written to %s" % (len(fp), a.write))
        return 0
    with open(a.reference) as f:
        ref = json.load(f)
    bad = sorted(k for k in set(ref) | set(fp) if ref.get(k) != fp.get(k))
    for k in bad:
        print("DIFFERS  %-36s reference %s  now %s" % (k, ref.get(k), fp.get(k)))
    print("%d/%d fingerprints identical" % (len(ref) - len(bad), len(ref)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
