"""Phase 1 milestone: inbound, outbound, relocation, dependencies and failure recovery on one small
deterministic scenario (basic map, explicit inventory, scripted orders, greedy FIFO coordinator).

    python experiments/checks/phase1.py [seed] [out_dir]

Orders
  O00000  t=0    inbound of two packages (3 kg: AGV alone; 9 kg: AGV + picking robot)
  O00001  t=0    outbound of two stored packages (two lines)
  O00002  t=0    relocate S0002 to L0005, then ship its package from the new location
  O00003  t=40   outbound of P00003; an idle robot breaks down on its rack cell for 90 ticks
                 -> stall (40 ticks), attempt failed, retry; recovery once the robot is repaired
                 (attempt 2 or 3, depending on how early the seed's AGV arrives)
  O00004  t=80   one package through all three operations: inbound -> relocate its shelf ->
                 outbound of the same package (three dependent tasks; one order)
  O00005  t=120  outbound of P00001, already shipped by O00001 -> fails without retry (unsatisfiable)

Gates (all must pass): G1 order outcomes, G2 inventory invariants every tick and conservation at
the end, G3 one package id across inbound/relocation/outbound, G4 dependencies (no dispatch before
the prerequisite completed), G5 failure record and recovery, G6 metrics from the event log
reconcile with the manager, G7 same seed -> identical event log, G8 every join and lift feasible.
Writes CSV ledgers to out_dir (default: orders_out/phase1, git-ignored).
"""
import os
import sys

import battery_tarware as bt
from battery_tarware.orders.reports import (episode_order_metrics, order_table, package_table, shelf_table,
                                            task_table)
from battery_tarware.orders.scenarios import order_scenario
from battery_tarware.robots import RobotStatus

OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join("orders_out", "phase1")
INITIAL = {"L0000": 1, "L0001": 1, "L0002": 1, "L0010": 0, "L0020": 1, "L0030": 0}
TRACE = [
    {"arrival": 0, "op": "inbound", "packages": [{"weight_kg": 3.0}, {"weight_kg": 9.0}]},
    {"arrival": 0, "op": "outbound", "packages": ["P00000", "P00001"], "due": 400},
    {"arrival": 0, "kind": "outbound", "tasks": [
        {"key": "move", "op": "relocation", "shelf": "S0002", "to": "L0005"},
        {"key": "ship", "op": "outbound", "package": "P00002", "depends_on": ["move"]}]},
    {"arrival": 40, "op": "outbound", "packages": ["P00003"]},
    {"arrival": 80, "kind": "mixed", "tasks": [
        {"key": "in", "op": "inbound", "weight_kg": 2.5},
        {"key": "mv", "op": "relocation", "shelf": "@in", "to": "L0012", "depends_on": ["in"]},
        {"key": "out", "op": "outbound", "package": "@in", "depends_on": ["mv"]}]},
    {"arrival": 120, "op": "outbound", "packages": ["P00001"]},
]
LIFECYCLE = {"stall_ticks": 40, "auto_retry": True, "retry_delay": 5, "max_dispatch_retries": 3}


def episode(seed):
    cfg = order_scenario(source={"kind": "trace", "orders": TRACE},
                         inventory={"initial": INITIAL, "package_weight": 4.0}, lifecycle=LIFECYCLE)
    env = bt.make(cfg, max_steps=3000)
    sim = env.unwrapped.sim
    env.reset(seed=seed)
    co = bt.GreedyCoalitionCoordinator("fifo")
    blocker = None
    while True:
        if sim.tick == 40:                                 # P00003's shelf S0004 stands on L0020
            blocker = [r for r in sim.robots if r.task_id is None and r.status == RobotStatus.IDLE
                       and r.spec.name == "picking_robot"][0]  # an uncommitted robot breaks down there
            blocker.cell = sim.orders.inventory.locations["L0020"].cell
            blocker.resume_status, blocker.status, blocker.repair_left = RobotStatus.IDLE, RobotStatus.FAILED, 90
            blocker.clear_path()
        _, _, term, trunc, _ = env.step(co.act(env))
        if term or trunc:
            break
    return env, sim, blocker.id


def main(seed=0):
    env, sim, blocker = episode(seed)
    m = sim.orders
    inv = m.inventory
    gates = {}
    states = {o.order_id: o.state for o in m.orders.values()}
    gates["G1 order outcomes"] = states == {"O00000": "completed", "O00001": "completed", "O00002": "completed",
                                            "O00003": "completed", "O00004": "completed", "O00005": "failed"}
    try:
        inv.check()
        c = inv.counts()
        gates["G2 inventory invariants + conservation"] = sum(c.values()) == inv.created == 7 and \
            c.get("shipped") == inv.shipped == 5 and c.get("stored") == 2
    except Exception as e:                                 # pragma: no cover
        gates["G2 inventory invariants + conservation"] = False
        print("inventory:", e)
    o4 = m.orders["O00004"]
    pid = m.tasks[o4.task_ids[0]].package_id
    hist = [h[1] for h in inv.packages[pid].history]
    gates["G3 one package id across operations"] = \
        [m.tasks[t].package_id for t in (o4.task_ids[0], o4.task_ids[2])] == [pid, pid] and \
        hist == ["arrived", "in_transit", "stored", "in_transit", "stored", "in_transit", "shipped"] and \
        inv.packages[pid].tasks == list(o4.task_ids[:1] + o4.task_ids[2:])
    dep_ok = True
    for t in m.tasks.values():
        for d in t.depends_on:
            if t.attempts and sim.tasks[t.attempts[0]].released < m.tasks[d].completed_at:
                dep_ok = False
    gates["G4 dependencies respected"] = dep_ok and m.tasks["T00005"].history[1][2] == "blocked"
    t3 = m.tasks[m.orders["O00003"].task_ids[0]]
    f = t3.failures[0] if t3.failures else None
    gates["G5 failure recorded + recovered"] = bool(f) and t3.attempt == len(t3.failures) + 1 and \
        t3.state.value == "completed" and [x.attempt for x in t3.failures] == list(range(1, t3.attempt)) and \
        all(("robot %d " % blocker) in (x.blocking_resource or "") and x.last_progress_tick is not None and
            x.snapshot and x.failure_reason.startswith("stall:") for x in t3.failures) and \
        m.tasks[m.orders["O00005"].task_ids[0]].failures[0].failure_reason == "inventory:package_P00001_shipped"
    met = episode_order_metrics(sim)
    gates["G6 event-log metrics reconcile"] = met["completed"] == 5 and met["failed"] == 1 and \
        met["orders_arrived"] == len(m.orders) and \
        met["tasks_completed"] == sum(t.state.value == "completed" for t in m.tasks.values()) and \
        met["retries"] == t3.attempt - 1 == met["stalls"]
    _, sim2, _ = episode(seed)
    key = lambda s: [(e.tick, e.type, e.task, sorted((k, str(v)) for k, v in e.data.items())) for e in s.events.events]
    gates["G7 deterministic"] = key(sim) == key(sim2)
    feas = True
    for e in sim.events.events:
        if e.type == "robot_joined" and not sim.tasks[e.task].type.role(e.data["role"]).accepts(sim.robots[e.robots[0]]):
            feas = False
        if e.type == "payload_picked" and not e.data.get("reload") and \
                sum(sim.robots[i].spec.payload_kg for i in e.robots) + 1e-9 < sim.tasks[e.task].weight_kg:
            feas = False
    gates["G8 joins and lifts feasible"] = feas

    print("seed %d: %d ticks, %d simulator tasks (attempts), events %d" % (seed, sim.tick, len(sim.tasks),
                                                                          len(sim.events)))
    print("\norders")
    for r in order_table(m):
        print("  %(order_id)s %(kind)-9s arrival %(arrival)4d  %(state)-9s closed %(closed)s  %(tasks)s  %(reason)s" % r)
    print("\ntasks")
    for r in task_table(m):
        print("  %(task_id)s %(order_id)s %(op)-10s %(state)-9s attempts %(attempt)d  pkg %(package_id)s  shelf "
              "%(shelf_id)s  dest %(dest_location)s  type %(task_type)s  deps [%(depends_on)s]  fail %(failure_reason)s"
              % r)
    print("\npackage %s history:" % pid)
    for h in inv.packages[pid].history:
        print("  tick %4d  %-10s %s" % h)
    print("\nmetrics:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in met.items()})
    print()
    for g, ok in gates.items():
        print("  %-42s %s" % (g, "PASS" if ok else "FAIL"))
    os.makedirs(OUT, exist_ok=True)
    for name, rows in (("orders", order_table(m)), ("tasks", task_table(m)), ("packages", package_table(m)),
                       ("shelves", shelf_table(m))):
        bt.write_csv(rows, os.path.join(OUT, "phase1_%s.csv" % name))
    print("\n%s" % ("ALL GATES PASS" if all(gates.values()) else "GATE FAILURE"))
    return all(gates.values())


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 0) else 1)
