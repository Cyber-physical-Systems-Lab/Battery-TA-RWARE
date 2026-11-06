"""Episode trace recorder (observational) and per-tick bottleneck attribution.

``TraceRecorder(env).update()`` after every step records what the event log does not hold: each
robot's cell, state of charge, status, task, role and **activity** per tick, the cells of shelves
that are off their rack location, and per tick the **cause** each open order-layer task is waiting
(or progressing) for. With the event log (``sim.events``) this reconstructs the full history; the
dashboard, the CSV ledgers and the metrics all read this one trace.

Robot activity (one per robot-tick)

==================  ===========================================================================
idle                no commitment
travel_empty        committed to a task and moved this tick, not carrying
travel_loaded       moved this tick carrying a payload
working             took part in stage work this tick
waiting_partner     at its work position but not working: the stage crew is incomplete or not all
                    present (coalition waiting)
waiting_blocked     committed, not at its work position and did not move (queue, congestion)
to_charger          heading to a dock
dock_queue          assigned to a dock, not charging and did not move (waiting for the dock)
charging            charging
failed / stranded   broken down / battery empty
==================  ===========================================================================

Order-task cause per tick (bottleneck attribution; every tick of every open task gets one)

=======================  ================================================================
blocked_dependency       blocked by a prerequisite task
blocked_inventory        blocked by an inventory condition (package, slot, location, shelf)
blocked_access           no route for a loaded carrier
capability               no operational robot (set) can fill a role or lift the payload
sequencing               feasible coalitions existed but the rule chose other tasks
coalition_unavailable    capable robots exist but are committed to other tasks
charging_contention      capable robots exist but are charging or heading to / queueing for docks
breakdown                capable robots (or a crew member) are broken down
battery                  feasible only without the battery reserve, or a crew member stranded
unreachable              no feasible route for the candidate crews
travel / execution       the crew is travelling / working
coalition_waiting        crew complete but a member waits at its position for the others
congestion               crew complete, a member neither at its position nor moving
failure_recovery         failed attempt waiting for its retry
unassigned               open, no scheduler decision recorded (coordinator without a decision log)
=======================  ================================================================
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np

from battery_tarware.orders.monitor import target_cell
from battery_tarware.robots import RobotStatus

ACTIVITIES = ("idle", "travel_empty", "travel_loaded", "working", "waiting_partner", "waiting_blocked",
              "to_charger", "dock_queue", "charging", "failed", "stranded")
STATUSES = ("idle", "task", "charge", "failed", "stranded")
CAUSES = ("blocked_dependency", "blocked_inventory", "blocked_access", "capability", "sequencing",
          "coalition_unavailable", "charging_contention", "breakdown", "battery", "unreachable", "travel",
          "execution", "coalition_waiting", "congestion", "failure_recovery", "unassigned")
DECISION_CAUSE = {"assigned": "sequencing", "robot_taken": "sequencing", "budget": "sequencing",
                  "infeasible:capability": "capability", "infeasible:payload": "capability",
                  "unavailable:robots_busy": "coalition_unavailable", "unavailable:charging": "charging_contention",
                  "unavailable:breakdown": "breakdown", "unavailable:no_free_robot": "coalition_unavailable",
                  "infeasible:battery": "battery", "infeasible:unreachable": "unreachable"}


def robot_activity(sim, r) -> str:
    if r.status == RobotStatus.STRANDED:
        return "stranded"
    if r.status == RobotStatus.FAILED:
        return "failed"
    if r.status == RobotStatus.CHARGE:
        if r.charging:
            return "charging"
        return "to_charger" if r.moved else "dock_queue"
    if r.task_id is None:
        return "idle"
    if r.worked:
        return "working"
    if r.moved:
        return "travel_loaded" if sim.carrying(r) else "travel_empty"
    t = sim.tasks[r.task_id]
    cell = target_cell(sim, t, r.id)
    if cell is not None and (r.cell == cell or sim._present(r, t, r.role, cell)):
        return "waiting_partner"
    return "waiting_blocked"


class TraceRecorder:
    def __init__(self, env_or_sim, scheduler=None, label: Optional[str] = None):
        s = env_or_sim
        self.sim = s.sim if hasattr(s, "sim") else s.unwrapped.sim if hasattr(s, "unwrapped") else s
        self.scheduler = scheduler
        self.label = label
        self.reset()

    def reset(self) -> None:
        self.initial = [[r.cell[0], r.cell[1], round(r.soc, 4)] for r in self.sim.robots]   # state before tick 1
        self.frames: List[List] = []           # per tick: per robot [y, x, soc, status, task, role, activity]
        self.shelves: List[List] = []          # per tick: [[shelf, y, x, status], ...] for shelves off their location
        self.series: Dict[str, List] = {k: [] for k in ("open_orders", "completed_orders", "open_tasks", "mean_soc")}
        self.spans: Dict[str, List[List]] = {}  # order task -> [[from_tick, to_tick, cause], ...] (to inclusive)
        self.cause_ticks: Dict[str, Dict[str, int]] = {}
        self.activity_ticks = np.zeros((len(self.sim.robots), len(ACTIVITIES)), dtype=np.int64)
        self._cursor = 0
        self._decision: Dict[int, str] = {}
        self._tick0 = self.sim.tick
        self._roles = sorted({role.name for tt in self.sim.config.task_types for role in tt.roles})

    # ---------------------------------------------------------------- per tick
    def update(self) -> None:
        sim = self.sim
        if sim.tick < self._tick0 + len(self.frames):          # the simulator was reset
            self.reset()
        for e in sim.events.since(self._cursor):
            if e.type == "schedule_decision":
                self._decision = {row[0]: row[5] for row in e.data["tasks"]}
        self._cursor = len(sim.events)
        frame = []
        for r in sim.robots:
            a = robot_activity(sim, r)
            self.activity_ticks[r.id, ACTIVITIES.index(a)] += 1
            frame.append([r.cell[0], r.cell[1], round(r.soc, 4), STATUSES.index(r.status.value),
                          -1 if r.task_id is None else r.task_id,
                          -1 if r.role is None else self._roles.index(r.role), ACTIVITIES.index(a)])
        self.frames.append(frame)
        self.series["mean_soc"].append(round(float(np.mean([r.soc for r in sim.robots])), 4))
        self.series["open_tasks"].append(sum(1 for t in sim.tasks.values() if not t.done))
        m = getattr(sim, "orders", None)
        if m is None:
            self.shelves.append([])
            self.series["open_orders"].append(None)
            self.series["completed_orders"].append(None)
            return
        moving = []
        for s in m.inventory.shelves.values():
            if s.status == "in_transit":
                ot = m.tasks.get(s.busy_task or "")
                t = sim.tasks[ot.sim_task] if ot is not None and ot.sim_task is not None else None
                if t is not None and t.carried and t.carriers:
                    c = sim.robots[min(t.carriers)].cell
                    moving.append([s.shelf_id, c[0], c[1], "in_transit"])
            elif s.status == "dropped":
                moving.append([s.shelf_id, s.cell[0], s.cell[1], "dropped"])
        self.shelves.append(moving)
        self.series["open_orders"].append(sum(1 for o in m.orders.values() if o.state == "open"))
        self.series["completed_orders"].append(sum(1 for o in m.orders.values() if o.state == "completed"))
        for ot in m.tasks.values():
            if ot.terminal:
                continue
            self._attribute(ot.task_id, self._cause(m, ot))

    def _attribute(self, tid: str, cause: str) -> None:
        tick = self.sim.tick
        sp = self.spans.setdefault(tid, [])
        if sp and sp[-1][2] == cause and sp[-1][1] == tick - 1:
            sp[-1][1] = tick
        else:
            sp.append([tick, tick, cause])
        d = self.cause_ticks.setdefault(tid, {})
        d[cause] = d.get(cause, 0) + 1

    def _cause(self, m, ot) -> str:
        sim = self.sim
        st = ot.state.value
        if st in ("pending", "blocked"):
            r = ot.reason or ""
            if r.startswith("dependency"):
                return "blocked_dependency"
            if r.startswith("inventory"):
                return "blocked_inventory"
            if r.startswith("infeasible"):
                return "capability"
            return "blocked_access" if r.startswith("access") else "blocked_dependency"
        if st in ("retrying", "failed"):
            return "failure_recovery"
        if ot.sim_task is None:
            return "unassigned"
        t = sim.tasks[ot.sim_task]
        crew = [sim.robots[i] for i in t.robots()]
        if any(r.status == RobotStatus.STRANDED for r in crew):
            return "battery"
        if any(r.status == RobotStatus.FAILED for r in crew):
            return "breakdown"
        if any(r.worked for r in crew):
            return "execution"
        if any(r.moved for r in crew):
            return "travel"
        if not crew or not sim._crew_assigned(t):
            d = self._decision.get(t.id)
            if d:
                return DECISION_CAUSE.get(d, "unassigned")
            if ot.ready_since is not None and ot.ready_since >= sim.tick - 1:
                return "sequencing"                # released after this tick's decision round
            return "unassigned"
        acts = [robot_activity(sim, r) for r in crew]
        return "congestion" if "waiting_blocked" in acts else "coalition_waiting"

    # ---------------------------------------------------------------- export
    def to_dict(self, metrics: Optional[Dict] = None, tables: Optional[Dict] = None) -> Dict:
        sim = self.sim
        lay = sim.layout
        zones = {z: [list(c) for c in lay.cells(z)] for z in lay.zones}
        meta = {
            "label": self.label, "scenario": sim.config.name, "seed": sim.seed, "model": sim.config.model,
            "ticks": len(self.frames), "first_tick": self._tick0 + 1, "seconds_per_tick": sim.config.seconds_per_tick,
            "shape": list(lay.shape), "zones": zones,
            "robots": [{"id": r.id, "name": r.name, "type": r.spec.name, "capabilities": list(r.spec.capabilities),
                        "payload_kg": r.spec.payload_kg, "capacity_wh": r.spec.energy.capacity_wh} for r in sim.robots],
            "chargers": [{"id": c.id, "cell": list(c.cell), "type": c.type, "power_w": c.power_w} for c in sim.chargers],
            "roles": self._roles, "statuses": list(STATUSES), "activities": list(ACTIVITIES), "causes": list(CAUSES),
            "rule": getattr(self.scheduler, "rule", None),
        }
        tasks = {t.id: {"type": t.type.name, "origin": list(t.origin), "destination": list(t.destination),
                        "stages": [[s.name, list(c)] for s, c in zip(t.type.stages, t.stage_cells)],
                        "weight_kg": round(t.weight_kg, 3), "released": t.released, "state": t.state.value}
                 for t in sim.tasks.values()}
        out = {"meta": meta, "initial": self.initial, "frames": self.frames, "shelves": self.shelves, "series": self.series,
               "spans": self.spans, "sim_tasks": tasks,
               "energy_wh": {r.name: round(r.energy_used_wh, 3) for r in sim.robots},
               "events": [[e.tick, e.type, list(e.robots), e.task, e.data] for e in sim.events.events]}
        if self.scheduler is not None:
            out["runtime"] = [[t, round(s * 1000, 4), n] for t, s, n in self.scheduler.runtime]
        if metrics is not None:
            out["metrics"] = metrics
        if tables is not None:
            out["tables"] = tables
        return out

    def save(self, path: str, **kw) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(**kw), f, default=_json_default, separators=(",", ":"))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray, tuple, set)):
        return list(o)
    if hasattr(o, "value"):
        return o.value
    return str(o)
