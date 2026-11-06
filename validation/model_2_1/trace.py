"""Residual-event trace (model 2.1 gate): what a flagged robot wanted, what blocked it, and why.

  python validation/model_2_1/trace.py <scenario> <seed> <config> <robot> <tick> <L1|L2>
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate                                         # noqa: E402
import battery_tarware as bt                        # noqa: E402
import battery_tarware.traffic as T                 # noqa: E402

sc, seed, cf, rid, tick, kind = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
start = tick - (100 if kind == "L1" else 14)
env = bt.make(gate.scenario(sc, cf))
sim = env.unwrapped.sim
co = bt.GreedyCoalitionCoordinator("fifo")
env.reset(seed=seed)
rows = []
orig = T.PIBT.step


def desc(o):
    t = sim.tasks.get(o.task_id) if o.task_id is not None else None
    s = "r%d %s %s %s" % (o.id, o.spec.name[:8], o.cell, o.status.value)
    if o.charging:
        s += " CHARGING"
    if not o.operational:
        s += " NONOPERATIONAL"
    if t is not None:
        crew = sim._stage_crew(t)
        s += " t%d/%s stage%d@%s carried=%s crew_assigned=%s crew_present=%s" % (
            t.id, o.role, t.stage, t.stage_cell, t.carried, sim._crew_assigned(t), crew is not None)
    return s


def step(pibt, units):
    orig(pibt, units)
    if sim.tick < start:
        return
    by = {u.id: u for u in units}
    u = by.get(rid)
    if u is None:
        return
    blockers = []
    if u.goal is not None and u.next == u.cell:
        d = pibt.route_distances(u)
        for dy, dx in T.MOVES:
            c = (u.cell[0] + dy, u.cell[1] + dx)
            if sim.layout.in_bounds(c) and d[c] < d[u.cell]:
                occ = [o for o in sim.robots if o.cell == c and o.id != rid]
                blockers.append((c, [desc(o) for o in occ] or ["free"]))
    rows.append((sim.tick, u.cell, u.goal, u.next, u.priority, u.fixed, u.lasting, blockers))
T.PIBT.step = step
while sim.tick < tick:
    env.step(co.act(env))
r = sim.robots[rid]
print("EVENT %s %s seed %d %s robot %d at tick %d" % (kind, sc, seed, cf, rid, tick))
print("  robot now:", desc(r))
print("  target now:", sim._target(r))
goals = collections.Counter(g for _, _, g, *_ in rows)
cells = collections.Counter(c for _, c, *_ in rows)
print("  goals over window:", dict(goals))
print("  cells over window:", dict(cells.most_common(6)))
moved = sum(1 for _, c, _, n, *_ in rows if n is not None and n != c)
print("  ticks %d, moved %d" % (len(rows), moved))
bl = collections.Counter()
for *_, b in rows:
    for c, occ in b:
        bl[(c, occ[0][:120])] += 1
for (c, o), n in bl.most_common(5):
    print("  blocker %s x%d: %s" % (c, n, o))
for row in rows[-4:]:
    print("  ", row[:7])
near = [o for o in sim.robots if abs(o.cell[0] - r.cell[0]) <= 2 and abs(o.cell[1] - r.cell[1]) <= 2 and o.id != rid]
for o in near:
    print("   near:", desc(o), "target", sim._target(o))
