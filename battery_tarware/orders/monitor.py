"""Progress monitoring and stall detection (read-only; usable on any simulator).

For every open task with a crew, each tick is classified:

* **progress**: stage work advanced, the payload moved, the crew changed, or a crew member got
  closer to its target cell than ever before in this crew configuration;
* **legitimate waiting** (the stall clock does not run): ``waiting_partner`` (the current stage's
  crew is incomplete), ``robot_repair`` / ``robot_stranded`` (a crew member is broken down or out of
  energy and being replaced);
* **no progress** (the stall clock runs): ``waiting_cell`` (a crew member's target cell is held by a
  robot outside the crew; the holder is recorded as the blocking resource) or ``no_progress``.

After ``stall_ticks`` clock ticks a ``StallReport`` is produced once (until progress resumes), with
the last progress tick, the blocking resource, the number of blocked moves of the crew, and a
snapshot of the task and crew. Tasks without a crew are not monitored: waiting to be assigned is
a sequencing delay, not a stall.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from battery_tarware.robots import RobotStatus

LEGITIMATE = frozenset({"waiting_partner", "robot_repair", "robot_stranded"})


@dataclass
class StallReport:
    sim_task: int
    tick: int
    last_progress_tick: int
    stalled_ticks: int
    wait_reason: str
    blocking_resource: Optional[str]
    blocked_moves: int
    snapshot: Dict


@dataclass
class _Progress:
    last: int
    sig: Tuple = ()
    best: Dict[int, int] = field(default_factory=dict)
    clock: int = 0
    blocked_moves: int = 0
    reported: bool = False
    reason: str = "progress"
    resource: Optional[str] = None
    waits: Dict[str, int] = field(default_factory=dict)      # ticks by classification


def target_cell(sim, t, robot_id: int):
    """Where a crew member must be for the task's next step."""
    role = t.role_of(robot_id)
    if role == t.type.carrier_role:
        return t.stage_cell if t.carried else t.payload_cell
    s = t.next_stage_for(role)
    return t.stage_cells[s] if s is not None else sim.robots[robot_id].cell


def snapshot(sim, t) -> Dict:
    return {
        "tick": sim.tick,
        "task": {"id": t.id, "type": t.type.name, "state": t.state.value, "stage": t.stage,
                 "stage_name": t.type.stages[t.stage].name if t.stage < len(t.type.stages) else None,
                 "progress": round(t.progress, 3), "carried": t.carried,
                 "payload_cell": t.payload_cell, "assigned": {k: list(v) for k, v in t.assigned.items()}},
        "crew": [{"robot": r.id, "name": r.name, "role": t.role_of(r.id), "cell": r.cell, "status": r.status.value,
                  "soc": round(r.soc, 4), "target": target_cell(sim, t, r.id)}
                 for r in (sim.robots[i] for i in t.robots())],
    }


class ProgressMonitor:
    def __init__(self, sim, stall_ticks: int = 100):
        self.sim = sim
        self.stall_ticks = stall_ticks
        self.reset()

    def reset(self) -> None:
        self.state: Dict[int, _Progress] = {}
        self._cursor = len(self.sim.events)
        self.reports: List[StallReport] = []

    def update(self, task_ids: Optional[Iterable[int]] = None) -> List[StallReport]:
        sim = self.sim
        blocked: Dict[int, int] = {}
        for e in sim.events.since(self._cursor):
            if e.type == "collision_blocked":
                for r in e.robots:
                    blocked[r] = blocked.get(r, 0) + 1
        self._cursor = len(sim.events)
        ids = list(task_ids) if task_ids is not None else [t.id for t in sim.tasks.values() if not t.done]
        out = []
        for tid in ids:
            t = sim.tasks[tid]
            if t.done:
                continue
            crew = t.robots()
            st = self.state.get(tid)
            if st is None:
                st = self.state[tid] = _Progress(sim.tick)
            if not crew:                                  # unassigned: a sequencing delay
                st.sig, st.best, st.clock, st.reported = (), {}, 0, False
                st.reason, st.resource = "unassigned", None
                st.last = sim.tick
                continue
            sig = (t.stage, round(t.progress, 6), t.carried, t.payload_cell, tuple(sorted(crew)))
            progress = sig != st.sig
            if progress:
                st.sig, st.best = sig, {}
            for i in crew:
                d = sim.manhattan(sim.robots[i].cell, target_cell(sim, t, i))
                if i not in st.best:
                    st.best[i] = d
                elif d < st.best[i]:
                    st.best[i] = d
                    progress = True
            if progress:
                st.last, st.clock, st.blocked_moves, st.reported = sim.tick, 0, 0, False
                st.reason, st.resource = "progress", None
                st.waits["progress"] = st.waits.get("progress", 0) + 1
                continue
            st.reason, st.resource = self._classify(t, crew)
            st.waits[st.reason] = st.waits.get(st.reason, 0) + 1
            if st.reason in LEGITIMATE:
                continue
            st.clock += 1
            st.blocked_moves += sum(blocked.get(i, 0) for i in crew)
            if st.clock >= self.stall_ticks and not st.reported:
                st.reported = True
                rep = StallReport(tid, sim.tick, st.last, st.clock, st.reason, st.resource, st.blocked_moves,
                                  snapshot(sim, t))
                out.append(rep)
                self.reports.append(rep)
        return out

    def _classify(self, t, crew: List[int]) -> Tuple[str, Optional[str]]:
        sim = self.sim
        for i in crew:
            r = sim.robots[i]
            if r.status == RobotStatus.FAILED:
                return "robot_repair", "robot %d (%s) broken down, %d repair ticks left" % (r.id, r.name, r.repair_left)
            if r.status == RobotStatus.STRANDED:
                return "robot_stranded", "robot %d (%s) out of energy" % (r.id, r.name)
        if not sim._crew_assigned(t):
            open_roles = {role: n for role, n in t.open_roles().items() if t.needs_role_at_stage(role)}
            return "waiting_partner", "open roles %s" % (open_roles or t.open_roles())
        occ = sim.robot_cell_map()
        for i in crew:
            cell = target_cell(sim, t, i)
            others = [o for o in occ.get(cell, []) if o not in crew]
            if others and sim.robots[i].cell != cell:
                o = sim.robots[others[0]]
                what = ("task %d" % o.task_id) if o.task_id is not None else \
                    ("charging" if o.charging else o.status.value)
                return "waiting_cell", "cell %s held by robot %d (%s, %s)" % (cell, o.id, o.name, what)
        return "no_progress", None

    def waits(self, task_id: int) -> Dict[str, int]:
        st = self.state.get(task_id)
        return dict(st.waits) if st else {}
