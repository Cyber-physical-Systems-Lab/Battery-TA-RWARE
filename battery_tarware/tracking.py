"""Package and robot tracking (read-only).

``PackageTracker`` follows every package (the payload of a task with a carrier role) and every robot
through an episode, from the simulator's state and event log. It never changes the simulation:
call ``update()`` after each step (or let ``env.step`` do it with ``track=True``) and read the
ledgers at any time.

Per package: id, task, task type, job, weight, rack (origin cell), goal (destination cell), status,
current location, the robot carrying it, every robot that handled it (role, joined, left), loaded
distance, and the ticks it was released, first assigned, picked up, delivered, completed, dropped.

Package status (derived each tick from the task):

=============  =================================================================================
waiting        released, no robot committed
assigned       crew (partly) committed, not picked up yet
loading        the first (pick-up) stage is being worked
in_transit     carried towards the delivery cell
unloading      the stage at the destination is being worked
returning      delivered, carried on to a later stage (e.g. back to the rack)
dropped        a carrier stranded or failed; the payload rests on ``location`` until a new crew
delivered      task completed, final stage at the destination
stored         task completed, final stage back at a storage/other cell (e.g. a returned shelf)
cancelled      task cancelled
=============  =================================================================================

Per robot: total, empty and loaded distance (cells), packages handled, current package and role,
battery, activity, ticks by activity.

With the order layer (``orders:`` scenario key) a carrier task is one trip of a shelf, not a
package: records are then numbered ``M00000`` (moves) so they never collide with the inventory's
persistent package ids, and carry the order, order-layer task, attempt, operation, shelf and
inventory package as extra columns. The package ledger across trips is the inventory
(``battery_tarware.orders.reports.package_table``).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from battery_tarware.robots import RobotStatus
from battery_tarware.tasks.lifecycle import TaskState

Cell = Tuple[int, int]


@dataclass
class Handling:
    robot: int
    role: str
    joined: int
    left: Optional[int] = None
    reason: Optional[str] = None


@dataclass
class PackageRecord:
    package_id: str
    task_id: int
    task_type: str
    job_id: Optional[int]
    weight_kg: float
    rack: Cell
    goal: Cell
    released: int
    status: str = "waiting"
    location: Optional[Cell] = None
    carried_by: Tuple[int, ...] = ()
    handlers: List[Handling] = field(default_factory=list)
    distance_loaded: int = 0
    first_assigned: Optional[int] = None
    picked: Optional[int] = None
    delivered: Optional[int] = None
    completed: Optional[int] = None
    drops: int = 0
    deadline: Optional[int] = None
    extra: Dict = field(default_factory=dict)        # order layer links (empty without it)

    def row(self) -> Dict:
        return {"package_id": self.package_id, "task_id": self.task_id, "task_type": self.task_type,
                "job_id": self.job_id, "weight_kg": round(self.weight_kg, 2), "status": self.status,
                "rack": self.rack, "goal": self.goal, "location": self.location,
                "carried_by": "|".join(map(str, self.carried_by)),
                "handlers": "|".join("%d:%s@%d-%s" % (h.robot, h.role, h.joined, "" if h.left is None else h.left)
                                     for h in self.handlers),
                "distance_loaded": self.distance_loaded, "released": self.released,
                "first_assigned": self.first_assigned, "picked": self.picked, "delivered": self.delivered,
                "completed": self.completed, "deadline": self.deadline, "drops": self.drops, **self.extra}


@dataclass
class RobotRecord:
    robot_id: int
    name: str
    type: str
    distance: int = 0
    distance_empty: int = 0
    distance_loaded: int = 0
    packages: List[str] = field(default_factory=list)
    package: Optional[str] = None
    role: Optional[str] = None
    soc: float = 0.0
    activity: str = "idle"
    ticks: Dict[str, int] = field(default_factory=dict)

    def row(self) -> Dict:
        return {"robot_id": self.robot_id, "name": self.name, "type": self.type, "activity": self.activity,
                "soc": round(self.soc, 3), "package": self.package, "role": self.role,
                "distance": self.distance, "distance_empty": self.distance_empty,
                "distance_loaded": self.distance_loaded, "packages_handled": len(self.packages),
                **{"ticks_" + k: v for k, v in sorted(self.ticks.items())}}


class PackageTracker:
    def __init__(self, env_or_sim):
        self.sim = env_or_sim.sim if hasattr(env_or_sim, "sim") else env_or_sim
        self.reset()

    def reset(self) -> None:
        sim = self.sim
        self.packages: Dict[str, PackageRecord] = {}
        self.by_task: Dict[int, str] = {}
        self._cursor = 0
        self._cells = {r.id: r.cell for r in sim.robots}
        self.robots: Dict[int, RobotRecord] = {
            r.id: RobotRecord(r.id, r.name, r.spec.name, soc=r.soc) for r in sim.robots}
        self._tick = sim.tick
        self._scan_new_tasks()

    # ---------------------------------------------------------------- update
    def update(self) -> None:
        """Advance the ledgers to the simulator's current tick (call after every step)."""
        sim = self.sim
        if sim.tick < self._tick:                       # the simulator was reset
            self.reset()
            return
        self._scan_new_tasks()
        for e in sim.events.since(self._cursor):
            self._event(e)
        self._cursor = len(sim.events)
        self._move_and_status()
        self._tick = sim.tick

    def _scan_new_tasks(self) -> None:
        orders = getattr(self.sim, "orders", None)
        for t in self.sim.tasks.values():
            if t.id in self.by_task or t.type.carrier_role is None:
                continue
            pid = ("P%05d" if orders is None else "M%05d") % len(self.packages)
            rack = t.payload_cell if t.payload_cell is not None else t.stage_cells[0]
            self.packages[pid] = PackageRecord(pid, t.id, t.type.name, t.job_id, t.weight_kg, rack, t.destination,
                                               t.released, location=rack, deadline=t.deadline)
            self.by_task[t.id] = pid
            ot = orders.task_for_sim(t.id) if orders is not None else None
            if ot is not None:
                self.packages[pid].extra = {"order_id": ot.order_id, "order_task": ot.task_id, "attempt": ot.attempt,
                                            "op": ot.op, "shelf_id": ot.shelf_id, "inventory_package": ot.package_id}

    def _event(self, e) -> None:
        pid = self.by_task.get(e.task) if e.task is not None else None
        p = self.packages.get(pid) if pid else None
        if p is None:
            return
        if e.type == "robot_joined":
            for rid in e.robots:
                p.handlers.append(Handling(rid, e.data.get("role", ""), e.tick))
                if p.first_assigned is None:
                    p.first_assigned = e.tick
                if pid not in self.robots[rid].packages:
                    self.robots[rid].packages.append(pid)
        elif e.type == "robot_released":
            for rid in e.robots:
                for h in reversed(p.handlers):
                    if h.robot == rid and h.left is None:
                        h.left, h.reason = e.tick, e.data.get("reason")
                        break
        elif e.type == "payload_picked" and p.picked is None:
            p.picked = e.tick
        elif e.type == "payload_dropped":
            p.drops += 1
        elif e.type == "stage_completed":
            t = self.sim.tasks[e.task]
            dest = [i for i, s in enumerate(t.type.stages) if s.at == "destination"]
            name = e.data.get("stage")
            if dest and p.delivered is None and t.type.stages[dest[0]].name == name:
                p.delivered = e.tick
        elif e.type == "task_completed":
            p.completed = e.tick
            for h in p.handlers:
                if h.left is None:
                    h.left, h.reason = e.tick, "task completed"

    def _move_and_status(self) -> None:
        sim = self.sim
        steps: Dict[int, int] = {}
        for r in sim.robots:
            rec = self.robots[r.id]
            prev = self._cells[r.id]
            step = abs(r.cell[0] - prev[0]) + abs(r.cell[1] - prev[1])
            steps[r.id] = step
            rec.distance += step
            if sim.carrying(r):
                rec.distance_loaded += step
            else:
                rec.distance_empty += step
            self._cells[r.id] = r.cell
            rec.soc = r.soc
            rec.activity = _activity(sim, r)
            rec.ticks[rec.activity] = rec.ticks.get(rec.activity, 0) + 1
            rec.package = self.by_task.get(r.task_id) if r.task_id is not None else None
            rec.role = r.role
        for p in self.packages.values():
            t = sim.tasks[p.task_id]
            carriers = tuple(t.carriers) if t.carried else ()
            if carriers:
                lead = min(carriers)                  # joint carriers move as one unit
                p.location = sim.robots[lead].cell
                p.distance_loaded += steps[lead]
            elif t.payload_cell is not None:
                p.location = t.payload_cell
            p.carried_by = carriers
            p.status = _package_status(t, p)


def _activity(sim, r) -> str:
    if r.status == RobotStatus.STRANDED:
        return "stranded"
    if r.status == RobotStatus.FAILED:
        return "failed"
    if r.status == RobotStatus.CHARGE:
        return "charging" if r.charging else "to_charger"
    if r.task_id is None:
        return "idle"
    if r.worked:
        return "working"
    if sim.carrying(r):
        return "carrying"
    return "travelling" if r.moved else "waiting"


def _package_status(t, p: PackageRecord) -> str:
    if t.state == TaskState.CANCELLED:
        return "cancelled"
    if t.state == TaskState.COMPLETED:
        last = t.type.stages[-1].at
        return "delivered" if last == "destination" else "stored"
    stage = t.type.stages[t.stage] if t.stage < len(t.type.stages) else None
    if not t.carried:
        if p.picked is not None:
            return "dropped"
        if not t.robots():
            return "waiting"
        return "loading" if t.progress > 0 else "assigned"
    if stage is not None and stage.at == "destination" and t.progress > 0:
        return "unloading"
    return "returning" if p.delivered is not None else "in_transit"


def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        open(path, "w").close()
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def package_table(tracker: PackageTracker) -> List[Dict]:
    return [p.row() for p in tracker.packages.values()]


def robot_table(tracker: PackageTracker) -> List[Dict]:
    return [r.row() for r in tracker.robots.values()]


def status_counts(tracker: PackageTracker) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in tracker.packages.values():
        out[p.status] = out.get(p.status, 0) + 1
    return out
