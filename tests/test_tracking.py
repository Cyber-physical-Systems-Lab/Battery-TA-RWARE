"""Package and robot tracking: read-only, consistent with the simulator."""
import pytest

import battery_tarware as bt
from battery_tarware.tasks.lifecycle import TaskState


def run(scenario, ticks, seed=0, track=True):
    env = bt.make(scenario, max_steps=ticks)
    sim = env.unwrapped.sim
    co = bt.GreedyCoalitionCoordinator()
    env.reset(seed=seed)
    tr = bt.PackageTracker(sim) if track else None
    cells = []
    snapshots = []
    while True:
        _, _, te, tr_, _ = env.step(co.act(env))
        if tr is not None:
            tr.update()
            snapshots.append({pid: (p.status, p.location, p.carried_by) for pid, p in tr.packages.items()})
        cells.append(tuple(r.cell for r in sim.robots))
        if te or tr_:
            break
    return sim, tr, cells, snapshots


@pytest.mark.parametrize("scenario", ["basic", "intermediate", "advanced"])
def test_tracking_does_not_change_the_run(scenario):
    assert run(scenario, 600, track=True)[2] == run(scenario, 600, track=False)[2]


@pytest.mark.parametrize("scenario", ["basic", "intermediate", "advanced"])
def test_ledgers_agree_with_the_simulator(scenario):
    sim, tr, _, snaps = run(scenario, 1500, seed=1)
    for r in sim.robots:
        rec = tr.robots[r.id]
        assert rec.distance == r.distance
        assert rec.distance_empty + rec.distance_loaded == rec.distance
        assert sum(rec.ticks.values()) == sim.tick
    carrier_tasks = [t for t in sim.tasks.values() if t.type.carrier_role]
    assert len(tr.packages) == len(carrier_tasks)
    for p in tr.packages.values():
        t = sim.tasks[p.task_id]
        assert (t.state == TaskState.COMPLETED) == (p.status in ("delivered", "stored"))
        if p.completed is not None:
            assert p.picked is not None and p.picked <= p.completed and p.completed == t.completed_at
        if p.delivered is not None:
            assert p.picked <= p.delivered
        if t.carried:
            assert p.location == sim.robots[min(t.carriers)].cell and p.carried_by == tuple(t.carriers)
        for h in p.handlers:
            assert h.left is None or h.left >= h.joined
        if p.status in ("waiting", "assigned", "loading"):
            assert p.distance_loaded == 0
    # loaded distance of all packages = loaded distance of the lead carriers
    assert sum(p.distance_loaded for p in tr.packages.values()) <= sum(r.distance_loaded for r in tr.robots.values())
    assert any(p.status in ("delivered", "stored") for p in tr.packages.values())
    # every package moved through statuses in order
    order = ["waiting", "assigned", "loading", "in_transit", "unloading", "returning", "delivered", "stored"]
    for pid in tr.packages:
        seq = [s[pid][0] for s in snaps if pid in s and s[pid][0] != "dropped"]
        idx = [order.index(x) for x in seq]
        assert all(b >= a or (order[a] in ("assigned", "loading") and order[b] == "waiting") for a, b in zip(idx, idx[1:])), (pid, seq)


def test_tables_and_csv(tmp_path):
    sim, tr, _, _ = run("basic", 400)
    rows = bt.package_table(tr)
    assert rows and {"package_id", "status", "rack", "goal", "location", "handlers", "distance_loaded"} <= set(rows[0])
    bt.write_csv(rows, str(tmp_path / "p.csv"))
    bt.write_csv(bt.robot_table(tr), str(tmp_path / "r.csv"))
    assert (tmp_path / "p.csv").read_text().count("\n") == len(rows) + 1
    assert sum(bt.status_counts(tr).values()) == len(rows)
