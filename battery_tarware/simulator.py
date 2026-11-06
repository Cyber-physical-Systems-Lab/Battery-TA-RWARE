"""The simulation engine. One ``step`` = one tick:

1. apply robot actions (macro: commit / charge / release; primitive: move / interact / release)
2. random breakdowns and repairs
3. movement: path following (macro) or commanded moves (primitive), joint carriers as one unit,
   simultaneous conflict resolution, replanning and yielding
4. stage work: a stage advances only while its whole crew is present and operational
5. charging and energy accounting (idle, driving, payload, handling, charging power)
6. new demand (backlog / Poisson / batch / trace; tasks or dependency-linked jobs)
7. rewards and termination

The engine knows nothing about how decisions are made: coordinators and learning agents act through
actions, or through :meth:`assign`, :meth:`release` and :meth:`send_to_charger`.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from battery_tarware.config import ConfigError, ScenarioConfig, TaskTypeSpec
from battery_tarware.events import EventLog
from battery_tarware.layout import Layout
from battery_tarware.navigation import MOVES, astar, resolve_moves
from battery_tarware.robots import Robot, RobotStatus
from battery_tarware.tasks.lifecycle import TaskState
from battery_tarware.tasks.runtime import Task
from battery_tarware.traffic import PIBT, Unit

Cell = Tuple[int, int]

# primitive actions
P_WAIT, P_UP, P_DOWN, P_LEFT, P_RIGHT, P_INTERACT, P_RELEASE = range(7)
PRIMITIVE_ACTIONS = ("wait", "up", "down", "left", "right", "interact", "release")


def _heading(a, b) -> int:
    d = (b[0] - a[0], b[1] - a[1])
    return MOVES.index(d) if d in MOVES else 1


@dataclass
class Charger:
    id: int
    cell: Cell
    type: str
    power_w: float
    occupant: Optional[int] = None
    reserved_by: Optional[int] = None          # model 2.1: robot the dock is reserved for


@dataclass
class StepOutcome:
    reward: float
    robot_rewards: np.ndarray
    terminated: bool
    truncated: bool
    events: list
    invalid_actions: int


@dataclass
class _Stream:
    index: int
    items: List[Tuple[str, int]] = field(default_factory=list)   # (kind, id): ("task", tid) | ("job", jid)
    released: int = 0
    trace_pos: int = 0


class Simulator:
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.layout = Layout.from_spec(config.map, config.zones)
        self.capabilities = config.capabilities
        self.type_names = tuple(t.name for t in config.robot_types)
        self.task_type_names = tuple(t.name for t in config.task_types)
        self._robot_specs = config.resolved_robot_specs()
        self.chargers_template = self._place_chargers()
        self._passable_cache: Dict[Tuple[str, ...], np.ndarray] = {}
        self._traffic = PIBT(self)
        self.K = config.observation.task_slots
        self.C = len(self.chargers_template)
        self.n_macro_actions = 1 + self.K + self.C + 1
        self.n_primitive_actions = len(PRIMITIVE_ACTIONS)
        aisle = self.layout.cells("aisle")
        if len(aisle) < len(self._robot_specs):
            raise ConfigError("map has %d aisle cells for %d robots" % (len(aisle), len(self._robot_specs)))
        for t in config.task_types:
            for z in (t.origin, t.destination):
                if not self.layout.cells(z):
                    raise ConfigError("task type %r: map has no %r cells" % (t.name, z))
        self.reset(0)

    # ───────────────────────────────────────── setup
    def _place_chargers(self) -> List[Charger]:
        docks = self.layout.cells("dock")
        types = self.config.charger_types
        total = sum(t.share for t in types)
        counts = {t.name: 0 for t in types}
        out = []
        for i, cell in enumerate(docks):
            # interleave types in proportion to their share (deterministic)
            t = min(types, key=lambda t: ((counts[t.name] + 1) / (t.share / total), types.index(t)))
            counts[t.name] += 1
            out.append(Charger(i, cell, t.name, t.power_w))
        return out

    def passable(self, zones: Sequence[str]) -> np.ndarray:
        key = tuple(sorted(zones))
        if key not in self._passable_cache:
            self._passable_cache[key] = self.layout.passable_mask(key)
        return self._passable_cache[key]

    def reset(self, seed: Optional[int] = None) -> None:
        ss = np.random.SeedSequence(seed)
        spawn, demand, tasks, fail, obs = (np.random.default_rng(s) for s in ss.spawn(5))
        self.rng = {"spawn": spawn, "demand": demand, "tasks": tasks, "fail": fail, "obs": obs}
        self.seed = seed
        self.tick = 0
        self._hot_tick = None
        self.events = EventLog()
        self.chargers = [Charger(c.id, c.cell, c.type, c.power_w) for c in self.chargers_template]
        aisle = self.layout.cells("aisle")
        picks = spawn.choice(len(aisle), size=len(self._robot_specs), replace=False)
        self.robots: List[Robot] = []
        for i, ((name, spec, (lo, hi)), k) in enumerate(zip(self._robot_specs, picks)):
            soc = lo if lo == hi else float(spawn.uniform(lo, hi))
            self.robots.append(Robot(i, name, spec, self.type_names.index(self._base_type(name)), aisle[int(k)],
                                     soc * spec.energy.capacity_wh))
        self.tasks: Dict[int, Task] = {}
        self.jobs: Dict[int, List[int]] = {}
        self._next_task = 0
        self._next_job = 0
        self.streams = [_Stream(i) for i in range(len(self.config.demand))]
        self.dropped_demand = 0
        self._stranded_seen: Set[int] = set()
        self._release_demand()
        self._last_cursor = 0

    def _base_type(self, robot_name: str) -> str:
        idx = [n for n, _, _ in self._robot_specs].index(robot_name)
        g_offsets = 0
        for g in self.config.fleet:
            if idx < g_offsets + g.count:
                return g.type
            g_offsets += g.count
        raise KeyError(robot_name)

    # ───────────────────────────────────────── demand
    def _busy_origins(self) -> Set[Cell]:
        return {t.origin for t in self.tasks.values() if not t.done}

    def _sample_cell(self, zone: str, avoid: Set[Cell]) -> Optional[Cell]:
        cells = [c for c in self.layout.cells(zone) if c not in avoid]
        if not cells:
            return None
        return cells[int(self.rng["tasks"].integers(len(cells)))]

    def _new_task(self, ttype: TaskTypeSpec, job_id=None, origin=None, destination=None, depends=()) -> Optional[Task]:
        rng = self.rng["tasks"]
        if origin is None:
            origin = self._sample_cell(ttype.origin, self._busy_origins())
            if origin is None:
                return None
        if destination is None:
            destination = self._sample_cell(ttype.destination, set())
        cells = []
        for s in ttype.stages:
            if s.at == "origin":
                cells.append(origin)
            elif s.at == "destination":
                cells.append(destination)
            else:
                cells.append(self._sample_cell(s.at[5:], set()))
        weight = max(0.0, ttype.weight.sample(rng))
        tid = self._next_task
        self._next_task += 1
        t = Task(tid, ttype, origin, destination, weight, self.tick, tuple(cells), tuple(depends),
                 None if ttype.deadline is None else self.tick + ttype.deadline, job_id)
        self.tasks[tid] = t
        self.events.emit(self.tick, "task_released", task=tid, type=ttype.name, weight_kg=round(weight, 3))
        if not depends:
            self._make_ready(t)
        return t

    def _make_ready(self, t: Task) -> None:
        t.record.transition(TaskState.READY, self.tick)
        self.events.emit(self.tick, "task_ready", task=t.id)

    def _release_item(self, name: str, stream: _Stream) -> bool:
        if name in self.task_type_names:
            t = self._new_task(self.config.task_type(name))
            if t is None:
                self.dropped_demand += 1
                return False
            stream.items.append(("task", t.id))
            return True
        job = self.config.job(name)
        jid = self._next_job
        order, placed, pending = [], set(), list(job.tasks)
        while pending:                                  # topological order (validated acyclic)
            for jt in list(pending):
                if set(jt.depends_on) <= placed:
                    order.append(jt)
                    pending.remove(jt)
                    placed.add(jt.key)
        made: Dict[str, Task] = {}
        for jt in order:
            def ref(r):
                if r is None:
                    return None
                k, _, part = r.partition(".")
                return getattr(made[k], part)
            t = self._new_task(self.config.task_type(jt.type), jid, ref(jt.origin_from), ref(jt.destination_from),
                               tuple(made[d].id for d in jt.depends_on))
            if t is None:                               # cannot place the job: roll back its tasks
                for m in made.values():
                    del self.tasks[m.id]
                self.dropped_demand += 1
                return False
            made[jt.key] = t
        self._next_job += 1
        self.jobs[jid] = [t.id for t in made.values()]
        stream.items.append(("job", jid))
        return True

    def _item_done(self, item) -> bool:
        kind, i = item
        if kind == "task":
            return self.tasks[i].done
        return all(self.tasks[t].done for t in self.jobs[i])

    def _pick(self, spec, rng) -> str:
        items = spec.items()
        names = list(items)
        if len(names) == 1:
            return names[0]
        w = np.array([items[n] for n in names], dtype=float)
        return names[int(rng.choice(len(names), p=w / w.sum()))]

    def _release_demand(self) -> None:
        rng = self.rng["demand"]
        for spec, st in zip(self.config.demand, self.streams):
            cap = spec.max_items
            n = 0
            if spec.kind == "backlog":
                outstanding = sum(not self._item_done(it) for it in st.items)
                n = max(0, spec.open - outstanding)
            elif spec.kind == "poisson":
                rate = spec.rate
                if spec.rates:
                    rate = [r for t, r in spec.rates if t <= self.tick][-1]
                n = int(rng.poisson(rate))
            elif spec.kind == "batch":
                n = spec.count if self.tick == 0 else 0
            elif spec.kind == "trace":
                while st.trace_pos < len(spec.trace) and spec.trace[st.trace_pos] <= self.tick:
                    st.trace_pos += 1
                    n += 1
            for _ in range(n):
                if cap is not None and st.released >= cap:
                    break
                if self._release_item(self._pick(spec, rng), st):
                    st.released += 1

    def demand_finished(self) -> bool:
        """True when no stream will release anything more and every released item is done."""
        for spec, st in zip(self.config.demand, self.streams):
            if spec.kind in ("backlog", "poisson") and (spec.max_items is None or st.released < spec.max_items):
                return False
            if spec.kind == "trace" and st.trace_pos < len(spec.trace):
                return False
            if not all(self._item_done(it) for it in st.items):
                return False
        return True

    # ───────────────────────────────────────── queries
    def open_tasks(self) -> List[Task]:
        """Released, not finished, dependencies met; in release order (this is the task-slot order)."""
        return [t for t in self.tasks.values() if not t.done and t.state != TaskState.PENDING]

    def task_slots(self) -> List[Optional[Task]]:
        ts = [t for t in self.open_tasks() if t.joinable_roles()]
        ts = ts[:self.K]
        return ts + [None] * (self.K - len(ts))

    def robot_cell_map(self) -> Dict[Cell, List[int]]:
        occ: Dict[Cell, List[int]] = {}
        for r in self.robots:
            occ.setdefault(r.cell, []).append(r.id)
        return occ

    @staticmethod
    def chebyshev(a: Cell, b: Cell) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    @staticmethod
    def manhattan(a: Cell, b: Cell) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def carrying(self, r: Robot) -> bool:
        if r.task_id is None:
            return False
        t = self.tasks[r.task_id]
        return t.carried and r.role == t.type.carrier_role

    # ───────────────────────────────────────── commitments
    def can_join(self, r: Robot, t: Task, role: Optional[str] = None) -> Tuple[bool, str]:
        if not r.operational:
            return False, "robot not operational"
        if t.done or t.state == TaskState.PENDING:
            return False, "task not open"
        if r.task_id == t.id:
            return False, "already in this task"
        if self.carrying(r):
            return False, "robot is carrying a payload"
        roles = [role] if role is not None else [x.name for x in t.type.roles]
        joinable = t.joinable_roles()
        for name in roles:
            if name not in joinable:
                continue
            try:
                rs = t.type.role(name)
            except KeyError:
                return False, "unknown role %r" % name
            if len(t.assigned[name]) >= rs.count or not r.can(rs.capability):
                continue
            if name == t.type.carrier_role:
                if r.spec.payload_kg <= 0:
                    continue
                cap_now = sum(self.robots[i].spec.payload_kg for i in t.assigned[name])
                left = rs.count - len(t.assigned[name]) - 1
                best_other = max([o.spec.payload_kg for o in self.robots if o.can(rs.capability) and o.id != r.id
                                  and o.operational] or [0.0])
                if cap_now + r.spec.payload_kg + left * best_other < t.weight_kg - 1e-9:
                    continue
            return True, name
        return False, "no open role this robot can fill"

    def assign(self, task_id: int, robot_id: int, role: Optional[str] = None) -> Tuple[bool, str]:
        """Commit a robot to a task role (first fitting role if ``role`` is None). A robot already
        committed elsewhere (and not carrying) is released from there first."""
        r, t = self.robots[robot_id], self.tasks[task_id]
        ok, why = self.can_join(r, t, role)
        if not ok:
            return False, why
        if r.status in (RobotStatus.TASK, RobotStatus.CHARGE):
            self.release(r.id)
        t.assigned[why].append(r.id)
        r.status, r.task_id, r.role = RobotStatus.TASK, t.id, why
        r.clear_path()
        if t.state == TaskState.READY:
            t.record.transition(TaskState.ASSIGNED, self.tick)
        self.events.emit(self.tick, "robot_joined", robots=(r.id,), task=t.id, role=why)
        return True, why

    def assign_coalition(self, task_id: int, assignment: Dict[str, Sequence[int]]) -> Tuple[bool, str]:
        """Commit a whole crew atomically (all or nothing)."""
        t = self.tasks[task_id]
        for role, ids in assignment.items():
            open_n = t.type.role(role).count - len(t.assigned[role])
            if len(ids) > open_n:
                return False, "role %r has only %d open slot(s)" % (role, open_n)
        done = []
        for role, ids in assignment.items():
            for rid in ids:
                ok, why = self.assign(task_id, rid, role)
                if not ok:
                    for d in done:
                        self.release(d)
                    return False, "robot %d: %s" % (rid, why)
                done.append(rid)
        return True, "ok"

    def release(self, robot_id: int, reason: str = "released") -> Tuple[bool, str]:
        r = self.robots[robot_id]
        if self.carrying(r) and r.status != RobotStatus.STRANDED and r.status != RobotStatus.FAILED:
            return False, "cannot release a robot that is carrying a payload"
        if r.status == RobotStatus.CHARGE or r.charger is not None:
            self._stop_charging(r, reason)
            r.status = RobotStatus.IDLE if r.operational else r.status
            return True, "ok"
        if r.task_id is None:
            return False, "robot has no commitment"
        t = self.tasks[r.task_id]
        if self.carrying(r):                            # failed / stranded carrier drops the payload
            t.carried = False
            t.payload_cell = r.cell
            self.events.emit(self.tick, "payload_dropped", robots=tuple(t.carriers), task=t.id, cell=r.cell)
        t.assigned[r.role].remove(r.id)
        self.events.emit(self.tick, "robot_released", robots=(r.id,), task=t.id, role=r.role, reason=reason)
        r.task_id, r.role = None, None
        if r.operational:
            r.status = RobotStatus.IDLE
        elif r.status == RobotStatus.FAILED:
            r.resume_status = RobotStatus.IDLE
        r.clear_path()
        if not t.done and not t.robots():
            if t.state == TaskState.ASSIGNED and t.stage == 0 and t.progress == 0:
                t.record.transition(TaskState.READY, self.tick)          # reassignment, not failure
            elif t.state in (TaskState.ACTIVE, TaskState.ASSIGNED):
                t.record.transition(TaskState.SUSPENDED, self.tick, reason="crew released")
                self.events.emit(self.tick, "task_suspended", task=t.id)
        return True, "ok"

    def can_charge(self, r: Robot, c: Charger) -> Tuple[bool, str]:
        if not r.operational:
            return False, "robot not operational"
        if self.carrying(r):
            return False, "robot is carrying a payload"
        if not r.compatible_with(c.type):
            return False, "charger type %r incompatible" % c.type
        if "dock" not in r.zones(False) and r.cell != c.cell:
            return False, "robot may not enter docks"
        if r.charger == c.id:
            return False, "already assigned to this charger"
        if self.config.model != "2.0":
            if c.reserved_by not in (None, r.id) or c.occupant not in (None, r.id):
                return False, "charger reserved or occupied by another robot"
            if any(o.cell == c.cell and o.id != r.id and not o.operational for o in self.robots):
                return False, "charger unavailable: a broken-down or stranded robot stands on it"
        return True, "ok"

    def charger_state(self, c: Charger) -> str:
        """Model 2.1 dock state: available, reserved, occupied or unavailable."""
        if c.occupant is not None:
            return "occupied"
        if c.reserved_by is not None:
            return "reserved"
        if any(o.cell == c.cell and not o.operational for o in self.robots):
            return "unavailable"
        return "available"

    def send_to_charger(self, robot_id: int, charger_id: int) -> Tuple[bool, str]:
        r, c = self.robots[robot_id], self.chargers[charger_id]
        ok, why = self.can_charge(r, c)
        if not ok:
            return False, why
        if r.task_id is not None or r.charger is not None:
            self.release(r.id, "going to charge")
        r.status, r.charger = RobotStatus.CHARGE, c.id
        if self.config.model != "2.0":
            c.reserved_by = r.id
        r.clear_path()
        return True, "ok"

    def _stop_charging(self, r: Robot, reason: str) -> None:
        if r.charger is not None:
            c = self.chargers[r.charger]
            if c.occupant == r.id:
                c.occupant = None
            if c.reserved_by == r.id:
                c.reserved_by = None
            if r.charging:
                self.events.emit(self.tick, "charge_stopped", robots=(r.id,), charger=c.id, soc=round(r.soc, 4), reason=reason)
        r.charging = False
        r.charger = None

    # ───────────────────────────────────────── action masks
    def macro_action_mask(self, r: Robot, slots: Optional[List[Optional[Task]]] = None) -> np.ndarray:
        slots = self.task_slots() if slots is None else slots
        m = np.zeros(self.n_macro_actions, dtype=np.int8)
        m[0] = 1
        if not r.operational:
            if r.status == RobotStatus.FAILED and r.task_id is not None:
                m[-1] = 1                       # a broken robot can be released (drops its payload)
            return m
        for k, t in enumerate(slots):
            if t is not None and self.can_join(r, t)[0] and self.task_visible(r, t):
                m[1 + k] = 1
        for c in self.chargers:
            if self.can_charge(r, c)[0]:
                m[1 + self.K + c.id] = 1
        if (r.task_id is not None or r.charger is not None) and not self.carrying(r):
            m[-1] = 1
        return m

    def task_visible(self, r: Robot, t: Task) -> bool:
        if self.config.observation.task_visibility == "global" or r.spec.sensing_range is None:
            return True
        return self.chebyshev(r.cell, t.stage_cell) <= r.spec.sensing_range

    def primitive_action_mask(self, r: Robot) -> np.ndarray:
        m = np.zeros(self.n_primitive_actions, dtype=np.int8)
        m[P_WAIT] = 1
        if not r.operational:
            return m
        carrying = self.carrying(r)
        if carrying:
            t = self.tasks[r.task_id]
            if r.id != min(t.carriers):
                return m                            # followers move with the group leader
            zones = self._group_zones(t)
        else:
            zones = r.zones(False)
        pas = self.passable(zones)
        for k, (a, (dy, dx)) in enumerate(zip((P_UP, P_DOWN, P_LEFT, P_RIGHT), MOVES)):
            n = (r.cell[0] + dy, r.cell[1] + dx)
            if self.layout.in_bounds(n) and pas[n] and self.layout.move_ok(r.cell, n, k):
                m[a] = 1
        m[P_INTERACT] = 1
        if (r.task_id is not None or r.charger is not None) and not carrying:
            m[P_RELEASE] = 1
        return m

    def action_masks(self) -> np.ndarray:
        if self.config.action_mode == "macro":
            slots = self.task_slots()
            return np.stack([self.macro_action_mask(r, slots) for r in self.robots])
        return np.stack([self.primitive_action_mask(r) for r in self.robots])

    # ───────────────────────────────────────── step
    def step(self, actions: Sequence[int]) -> StepOutcome:
        actions = [int(a) for a in actions]
        if len(actions) != len(self.robots):
            raise ValueError("expected %d actions, got %d" % (len(self.robots), len(actions)))
        cursor = len(self.events)
        energy_before = np.array([r.energy_used_wh for r in self.robots])
        self.tick += 1
        for r in self.robots:
            r.moved = r.worked = False
        rewards = np.zeros(len(self.robots))
        invalid = self._apply_actions(actions)
        self._failures()
        if self.config.action_mode == "macro":
            self._pibt_move()
        else:
            self._move(self._primitive_intents(actions))
        self._work(rewards)
        self._charging_and_energy()
        self._release_demand()
        # rewards
        R = self.config.rewards
        used_kwh = (np.array([r.energy_used_wh for r in self.robots]) - energy_before) / 1000.0
        rewards += R.energy_per_kwh * used_kwh + R.per_tick
        for r in self.robots:
            if r.status == RobotStatus.STRANDED and r.id not in self._stranded_seen:
                self._stranded_seen.add(r.id)
                rewards[r.id] += R.stranded
        terminated = all(r.status == RobotStatus.STRANDED for r in self.robots) or self.demand_finished()
        truncated = not terminated and self.tick >= self.config.max_steps
        return StepOutcome(float(rewards.sum()), rewards, terminated, truncated, self.events.since(cursor), invalid)

    # ---- 1. actions
    def _apply_actions(self, actions: List[int]) -> int:
        invalid = 0
        if self.config.action_mode == "macro":
            for r, a in zip(self.robots, actions):
                if a == 0:
                    continue
                if not 0 <= a < self.n_macro_actions:
                    invalid += 1
                    continue
                slots = self.task_slots()             # recomputed: earlier joins change the board
                if a <= self.K:
                    t = slots[a - 1]
                    ok = t is not None and self.task_visible(r, t) and self.assign(t.id, r.id)[0]
                elif a <= self.K + self.C:
                    ok = self.send_to_charger(r.id, a - 1 - self.K)[0]
                else:
                    ok = self.release(r.id)[0]
                invalid += not ok
        else:
            for r, a in zip(self.robots, actions):
                if not 0 <= a < self.n_primitive_actions:
                    invalid += 1
                elif a == P_INTERACT:
                    invalid += not self._interact(r)
                elif a == P_RELEASE:
                    invalid += not self.release(r.id)[0]
        return invalid

    def _interact(self, r: Robot) -> bool:
        """Primitive 'interact': dock and charge if on a charger, otherwise join a task whose current
        (or next needed) stage this robot is already positioned for."""
        if not r.operational or self.carrying(r):
            return False
        for c in self.chargers:
            if c.cell == r.cell:
                return self.send_to_charger(r.id, c.id)[0]
        if r.task_id is not None:
            return False
        for t in self.open_tasks():
            ok, role = self.can_join(r, t)
            if not ok:
                continue
            target = t.payload_cell if (role == t.type.carrier_role and not t.carried) else None
            stage = t.next_stage_for(role)
            if stage is None:
                continue
            cell = target or t.stage_cells[stage]
            if self._present(r, t, role, cell):
                return self.assign(t.id, r.id, role)[0]
        return False

    # ---- 2. failures
    def _failures(self) -> None:
        rng = self.rng["fail"]
        for r in self.robots:
            if r.status == RobotStatus.FAILED:
                r.repair_left -= 1
                if r.repair_left <= 0:
                    r.status = r.resume_status or RobotStatus.IDLE
                    r.resume_status = None
                    if r.status == RobotStatus.TASK and r.task_id is None:
                        r.status = RobotStatus.IDLE
                    self.events.emit(self.tick, "robot_repaired", robots=(r.id,))
                continue
            if r.status == RobotStatus.STRANDED or r.spec.failure.rate <= 0:
                continue
            if rng.random() < r.spec.failure.rate:
                if r.charging or (self.config.model != "2.0" and r.charger is not None):
                    self._stop_charging(r, "failure")
                r.resume_status = RobotStatus.TASK if r.task_id is not None else RobotStatus.IDLE
                r.status = RobotStatus.FAILED
                r.repair_left = r.spec.failure.repair_ticks
                r.clear_path()
                self.events.emit(self.tick, "robot_failed", robots=(r.id,), task=r.task_id)

    # ---- 3. movement
    def _group_zones(self, t: Task) -> Tuple[str, ...]:
        zs = None
        for i in t.carriers:
            z = set(self.robots[i].zones(True))
            zs = z if zs is None else zs & z
        return tuple(sorted(zs or ()))

    def _present(self, r: Robot, t: Task, role: str, cell: Cell) -> bool:
        rs = t.type.role(role)
        if role == t.type.carrier_role or rs.presence == "on_cell":
            return r.cell == cell
        return self.chebyshev(r.cell, cell) <= rs.radius

    def _target(self, r: Robot) -> Optional[Cell]:
        """Where a macro-controlled robot wants to be (None = stay)."""
        if r.status == RobotStatus.CHARGE:
            cell = self.chargers[r.charger].cell
            if self.config.model != "2.0" and r.cell != cell and any(
                    o.cell == cell and o.id != r.id and self._stationed(o) for o in self.robots):
                return self._staging_cell(r, cell)          # model 2.1: wait off the approach
            return cell
        if r.status == RobotStatus.TASK:
            t = self.tasks[r.task_id]
            if r.role == t.type.carrier_role and not t.carried:
                return self._queue_if_taken(r, t, t.payload_cell, r.zones(False))
            s = t.next_stage_for(r.role)
            if s is None:
                return None
            cell = t.stage_cells[s]
            rs = t.type.role(r.role)
            if rs.presence == "on_cell":
                return self._queue_if_taken(r, t, cell, r.zones(False))
            if t.type.carrier_role is not None:
                # helpers wait at a wide cell nearby until the carrier stands on the stage cell, so
                # 1-wide aisles are only ever passed through, never blocked by waiting robots
                carriers = t.carriers
                need = t.type.role(t.type.carrier_role).count
                ready = s == t.stage and len(carriers) == need and all(self.robots[i].cell == cell for i in carriers)
                if not ready:
                    return self._staging_cell(r, cell)
            if self.chebyshev(r.cell, cell) <= rs.radius:
                return r.cell
            return self._nearest_standing_cell(r, cell, rs.radius)
        if r.status == RobotStatus.IDLE and not self._parking(r)[r.cell]:
            return self._nearest_free_aisle(r)               # park out of the way
        return None

    def _parking(self, r: Robot) -> np.ndarray:
        """Cells where an idle robot may wait: wide aisle cells (>= 3 passable neighbours) at least
        two cells from any goal or custom work zone (hotspots where crews have to stand)."""
        key = tuple(sorted(r.zones(False)))
        cache = self.__dict__.setdefault("_parking_cache", {})
        if key not in cache:
            pas = self.passable(key)
            H, W = self.layout.shape
            hot = np.zeros((H, W), bool)
            for z in self.layout.zones:
                if z == "goal" or z not in ("aisle", "storage", "dock"):
                    for (y, x) in self.layout.cells(z):
                        hot[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
            aisle = self.layout.passable_mask(("aisle",))
            deg = np.zeros((H, W), int)
            for dy, dx in MOVES:
                shifted = np.zeros((H, W), bool)
                ys, yd = (slice(max(0, dy), H + min(0, dy)), slice(max(0, -dy), H - max(0, dy)))
                xs, xd = (slice(max(0, dx), W + min(0, dx)), slice(max(0, -dx), W - max(0, dx)))
                shifted[yd, xd] = pas[ys, xs]
                deg += shifted
            park = aisle & pas & (deg >= 3) & ~hot
            if not park.any():                                # tiny maps: fall back to wide aisle cells
                park = aisle & pas & (deg >= 3)
            cache[key] = park
        return cache[key]

    def _queue_if_taken(self, r: Robot, t: Task, cell: Cell, zones) -> Cell:
        """Head for ``cell``, unless a robot of another task holds it: then wait at a wide cell
        nearby (a station queue), so robots bound for a busy cell do not clog its approaches."""
        v21 = self.config.model != "2.0"
        if r.cell == cell:
            # model 2.1: a robot holding a shared work cell while its own crew is incomplete gives
            # it up to another task whose crew is complete (no hold-and-wait on shared cells)
            if v21 and not self._crew_assigned(t) and self._cell_wanted(t, cell, complete=True):
                return self._staging_cell(r, cell, zones)
            return cell
        if any(o.cell == cell and o.task_id != t.id and self._stationed(o) for o in self.robots):
            return self._staging_cell(r, cell, zones)
        if v21 and not self._crew_assigned(t) and self._cell_wanted(t, cell, complete=False):
            # model 2.1 admission: do not occupy a stage cell another task also needs until this
            # stage's crew is complete
            return self._staging_cell(r, cell, zones)
        return cell

    def _cell_wanted(self, t: Task, cell: Cell, complete: bool) -> bool:
        """Does another open, staffed task need ``cell`` for its current stage (with a complete crew
        for that stage, if ``complete``)?"""
        for u in self.tasks.values():
            if u.id != t.id and not u.done and u.robots() and u.stage < len(u.stage_cells) \
                    and u.stage_cells[u.stage] == cell and (not complete or self._crew_assigned(u)):
                return True
        return False

    def _crew_assigned(self, t: Task) -> bool:
        """Every role of the task's current stage has its full count of robots assigned."""
        if t.stage >= len(t.type.stages):
            return True
        return all(len(t.assigned[role]) >= t.type.role(role).count for role in t.stage_spec.roles)

    def _stationed(self, o: Robot) -> bool:
        """Is ``o`` staying where it is (as opposed to passing through)?"""
        if not o.operational or o.charging:
            return True
        if o.task_id is None:
            return False
        t = self.tasks[o.task_id]
        if o.role == t.type.carrier_role:
            return o.cell == (t.stage_cell if t.carried else t.payload_cell)
        return t.type.role(o.role).presence == "on_cell" and o.cell == t.stage_cell

    def _staging_cell(self, r: Robot, near: Cell, zones=None) -> Cell:
        zones = r.zones(False) if zones is None else zones
        neutral = self._neutral_cell if self.config.model != "2.0" else (lambda c: True)
        if self._degree(r, r.cell, zones) >= 3 and 1 < self.manhattan(r.cell, near) <= 4 and neutral(r.cell):
            return r.cell
        pas = self.passable(zones)
        occ = self.robot_cell_map()
        H, W = self.layout.shape
        best, best_key = r.cell, None
        for y in range(max(0, near[0] - 6), min(H, near[0] + 7)):
            for x in range(max(0, near[1] - 6), min(W, near[1] + 7)):
                c = (y, x)
                if not pas[c] or self._degree(r, c, zones) < 3 or self.chebyshev(c, near) <= 1 \
                        or any(o != r.id for o in occ.get(c, [])) or not neutral(c):
                    continue
                key = (self.manhattan(c, near), self.manhattan(c, r.cell), c)
                if best_key is None or key < best_key:
                    best, best_key = c, key
        return best

    def _neutral_cell(self, c: Cell) -> bool:
        """Model 2.1: a cell a robot may wait on: plain aisle or storage, and not the current stage
        cell or payload cell of an open task (waiting there would block another crew's work)."""
        if self.layout.zone_of(c) not in ("aisle", "storage"):
            return False
        if self.__dict__.get("_hot_tick") != self.tick:
            hot = set()
            for t in self.tasks.values():
                if not t.done and t.robots():
                    if t.stage < len(t.stage_cells):
                        hot.add(t.stage_cells[t.stage])
                    if t.payload_cell is not None:
                        hot.add(t.payload_cell)
            self._hot, self._hot_tick = hot, self.tick
        return c not in self._hot

    def _degree(self, r: Robot, cell: Cell, zones=None) -> int:
        pas = self.passable(r.zones(False) if zones is None else zones)
        return sum(1 for dy, dx in MOVES if self.layout.in_bounds((cell[0] + dy, cell[1] + dx))
                   and pas[cell[0] + dy, cell[1] + dx])

    def _nearest_standing_cell(self, r: Robot, center: Cell, radius: int) -> Cell:
        """Where a helper stands next to a stage cell: the nearest candidate within ``radius`` that it
        can actually reach with stationed robots treated as obstacles (a teammate standing on a
        crossing must not cut it off), preferring cells that keep the stage cell's access free.
        The choice is kept while it stays valid, so helpers do not oscillate between cells."""
        pas = self.passable(r.zones(False))
        stationed = {o.cell for o in self.robots if o.id != r.id and self._stationed_or_helping(o)}
        keep = r.path_goal
        if keep is not None and self.chebyshev(keep, center) <= radius and keep != center \
                and pas[keep] and keep not in stationed:
            return keep
        cands = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                c = (center[0] + dy, center[1] + dx)
                if not self.layout.in_bounds(c) or not pas[c] or c in stationed:
                    continue
                in_the_way = (dy == 0 and dx == 0) or abs(dy) + abs(dx) == 1
                cands.append((in_the_way, self.manhattan(r.cell, c), c))
        for _, _, c in sorted(cands):
            if c == r.cell or astar(pas, r.cell, c, stationed - {c}, self.layout.directions) is not None:
                r.path_goal = c
                return c
        return sorted(cands)[0][2] if cands else center

    def _stationed_or_helping(self, o: Robot) -> bool:
        if self._stationed(o):
            return True
        if o.task_id is None:
            return False
        t = self.tasks[o.task_id]
        rs = t.type.role(o.role)
        return (o.role != t.type.carrier_role and rs.presence == "adjacent"
                and self.chebyshev(o.cell, t.stage_cell) <= rs.radius and o.role in t.stage_spec.roles)

    def _nearest_free_aisle(self, r: Robot) -> Optional[Cell]:
        occ = self.robot_cell_map()
        park = self._parking(r)
        cells = [c for c in self.layout.cells("aisle") if not occ.get(c) or occ[c] == [r.id]]
        if not cells:
            return None
        return min(cells, key=lambda c: (not park[c], self.manhattan(r.cell, c), c))

    def _stage_crew(self, t: Task) -> Optional[List[Robot]]:
        """The crew of the current stage if all of it is present and operational, else None."""
        if t.done or t.state == TaskState.PENDING or t.stage >= len(t.type.stages):
            return None
        if t.type.carrier_role and not t.carried and t.stage > 0:
            return None                                       # payload must be re-picked first
        crew: List[Robot] = []
        for role in t.stage_spec.roles:
            ids = t.assigned[role]
            if len(ids) < t.type.role(role).count:
                return None
            for i in ids:
                rob = self.robots[i]
                if not rob.operational or not self._present(rob, t, role, t.stage_cell):
                    return None
                crew.append(rob)
        return crew

    def _pibt_move(self) -> None:
        working = {r.id for t in self.tasks.values() for r in (self._stage_crew(t) or [])}
        followers = set()
        units: List[Unit] = []
        for t in self.tasks.values():
            if t.carried and not t.done:
                followers.update(sorted(t.carriers)[1:])
        for r in self.robots:
            if r.id in followers:
                continue
            if self.carrying(r):
                t = self.tasks[r.task_id]
                members = [self.robots[i] for i in t.carriers]
                ok = all(m.operational for m in members)
                if ok:
                    t.move_credit = min(1.0, t.move_credit + min(m.spec.speed for m in members))
                zones = self._group_zones(t)
                goal = self._queue_if_taken(r, t, t.stage_cell, zones) if ok else None
                credit, key = t.move_credit, ("task", t.id)
            else:
                members = [r]
                ok = r.operational
                if ok:
                    r.move_credit = min(1.0, r.move_credit + r.spec.speed)
                goal = self._target(r) if ok else None
                zones = tuple(sorted(r.zones(False)))
                credit = r.move_credit
                key = ("task", r.task_id) if r.task_id is not None and r.role == self.tasks[r.task_id].type.carrier_role else None
            if goal is not None and goal != r.cell:
                r.elapsed += 1
            else:
                r.elapsed = 0
            fixed = (not ok or any(m.charging for m in members) or any(m.id in working for m in members)
                     or credit < 1.0 - 1e-9)
            lasting = not ok or any(m.charging for m in members)     # charging, failed or stranded
            if self.config.model != "2.0" and not lasting and goal == r.cell and self._stationed(r):
                lasting = True       # model 2.1: parked on its own stage cell until the stage is done
            meet = None
            if self.config.model != "2.0" and r.task_id is not None:
                t = self.tasks[r.task_id]
                if r.role == t.type.carrier_role:
                    meet = (t.id, t.stage_cell if t.carried else t.payload_cell)
                else:
                    mc = self._meet_cell(r)
                    meet = None if mc is None else (t.id, mc)
            units.append(Unit(r.id, members, r.cell, goal, tuple(sorted(zones)), key, fixed, r.elapsed, lasting, meet))
        self._traffic.step(units)
        for u in units:
            lead = self.robots[u.id]
            if u.next is not None and u.next != u.cell:
                for m in u.members:
                    m.heading = _heading(m.cell, u.next)
                    m.cell, m.moved = u.next, True
                    m.distance += 1
                if self.carrying(lead):
                    self.tasks[lead.task_id].move_credit -= 1.0
                else:
                    lead.move_credit -= 1.0
                lead.blocked_ticks = 0
            elif u.goal is not None and u.goal != u.cell and not u.fixed:
                lead.blocked_ticks += 1
                if lead.blocked_ticks == 1:
                    self.events.emit(self.tick, "collision_blocked", robots=(lead.id,), cell=u.cell)

    def _primitive_intents(self, actions: List[int]) -> Dict[int, Cell]:
        intents: Dict[int, Cell] = {}
        dirs = {P_UP: MOVES[0], P_DOWN: MOVES[1], P_LEFT: MOVES[2], P_RIGHT: MOVES[3]}
        for r, a in zip(self.robots, actions):
            if a not in dirs or not r.operational:
                continue
            if not self.primitive_action_mask(r)[a]:
                continue
            if r.charging:
                self._stop_charging(r, "moved")
                r.status = RobotStatus.IDLE
            if self.carrying(r):
                t = self.tasks[r.task_id]
                members = [self.robots[i] for i in t.carriers]
                if not all(m.operational for m in members):
                    continue
                t.move_credit = min(1.0, t.move_credit + min(m.spec.speed for m in members))
                if t.move_credit < 1.0 - 1e-9:
                    continue
            else:
                r.move_credit = min(1.0, r.move_credit + r.spec.speed)
                if r.move_credit < 1.0 - 1e-9:
                    continue
            dy, dx = dirs[a]
            intents[r.id] = (r.cell[0] + dy, r.cell[1] + dx)
        return intents

    def _share_key(self, r: Robot) -> int:
        """Robots with equal keys may share a cell: carriers of one task at its payload (co-loading
        and joint transport)."""
        if r.task_id is not None and r.role == self.tasks[r.task_id].type.carrier_role:
            return 10_000_000 + r.task_id
        return -1 - r.id

    def _meet_cell(self, r: Robot) -> Optional[Cell]:
        """Model 2.1: the cell where a non-carrier crew member with ``on_cell`` presence works beside
        its task's carrier (the two may share it), or None."""
        if r.task_id is None:
            return None
        t = self.tasks[r.task_id]
        if t.type.carrier_role is None or r.role == t.type.carrier_role:
            return None
        if t.type.role(r.role).presence != "on_cell":
            return None
        s = t.next_stage_for(r.role)
        return None if s is None else t.stage_cells[s]

    def _move(self, intents: Dict[int, Cell]) -> None:
        # one mover per robot, except joint carrier groups (lead only; followers ride along)
        followers: Dict[int, List[int]] = {}
        movers: Dict[int, Cell] = {}
        for t in self.tasks.values():
            if t.carried and not t.done:
                lead = min(t.carriers)
                followers[lead] = [i for i in t.carriers if i != lead]
        skip = {f for fs in followers.values() for f in fs}
        positions = {r.id: r.cell for r in self.robots if r.id not in skip}
        shared = {r.id: self._share_key(r) for r in self.robots if r.id not in skip}
        if self.config.model != "2.0":                  # crew co-location on the stage cell
            for r in self.robots:
                cell = self._meet_cell(r)
                if r.id in shared and cell is not None and intents.get(r.id, r.cell) == cell:
                    shared[r.id] = 10_000_000 + r.task_id
        intents = {m: c for m, c in intents.items() if m in positions}
        ok = resolve_moves(positions, intents, shared)
        for m, cell in intents.items():
            group = [m] + followers.get(m, [])
            lead = self.robots[m]
            if m in ok:
                for i in group:
                    rr = self.robots[i]
                    rr.heading = _heading(rr.cell, cell)
                    rr.cell = cell
                    rr.moved = True
                    rr.distance += 1
                if lead.path and lead.path[0] == cell:
                    lead.path.pop(0)
                if self.carrying(lead):
                    self.tasks[lead.task_id].move_credit -= 1.0
                else:
                    lead.move_credit -= 1.0
                lead.blocked_ticks = 0
            else:
                lead.blocked_ticks += 1
                if lead.blocked_ticks == 1:
                    self.events.emit(self.tick, "collision_blocked", robots=(m,), cell=cell)

    # ---- 4. stage work
    def _work(self, rewards: np.ndarray) -> None:
        R = self.config.rewards
        for t in list(self.tasks.values()):
            if t.done or t.state == TaskState.PENDING:
                continue
            # re-pick a dropped payload (after a carrier failure / battery loss) before continuing
            if t.type.carrier_role and not t.carried and t.stage > 0:
                cr = t.carriers
                need = t.type.role(t.type.carrier_role).count
                if len(cr) == need and all(self.robots[i].operational and self.robots[i].cell == t.payload_cell for i in cr):
                    t.carried, t.payload_cell = True, None
                    self.events.emit(self.tick, "payload_picked", robots=tuple(cr), task=t.id, reload=True)
                else:
                    continue
            spec = t.stage_spec
            crew = self._stage_crew(t)
            if crew is None:
                continue
            if t.progress == 0:
                if t.state in (TaskState.ASSIGNED, TaskState.SUSPENDED):
                    t.record.transition(TaskState.ACTIVE, self.tick)
                self.events.emit(self.tick, "stage_started", robots=tuple(r.id for r in crew), task=t.id,
                                 stage=spec.name)
            t.progress += min(r.spec.handling_speed for r in crew)
            for r in crew:
                r.worked = True
            if t.progress + 1e-9 < spec.duration:
                continue
            # stage complete
            self.events.emit(self.tick, "stage_completed", robots=tuple(r.id for r in crew), task=t.id, stage=spec.name)
            for r in crew:
                rewards[r.id] += R.stage_completed / len(crew)
            if t.stage == 0 and t.type.carrier_role and not t.carried:
                t.carried, t.payload_cell = True, None
                self.events.emit(self.tick, "payload_picked", robots=tuple(t.carriers), task=t.id)
            t.stage += 1
            t.progress = 0.0
            if t.stage == len(t.type.stages):
                self._complete(t, rewards)
            else:
                nxt = t.type.stages[t.stage].roles
                for role in list(t.assigned):
                    if role == t.type.carrier_role:
                        continue
                    done_role = t.next_stage_for(role) is None
                    if done_role or (role not in nxt and not t.type.role(role).stay):
                        for i in list(t.assigned[role]):     # freed until (if ever) needed again
                            self.release(i, "role finished" if done_role else "not needed until a later stage")

    def _complete(self, t: Task, rewards: np.ndarray) -> None:
        R = self.config.rewards
        if t.carried:
            t.carried, t.payload_cell = False, t.stage_cells[-1]
            self.events.emit(self.tick, "payload_placed", robots=tuple(t.carriers), task=t.id, cell=t.payload_cell)
        crew = t.robots()
        late = 0 if t.deadline is None else max(0, self.tick - t.deadline)
        value = R.task_completed * t.type.priority + R.late_per_tick * late
        for i in crew:
            rewards[i] += value / max(1, len(crew))
        t.completed_at = self.tick
        t.record.transition(TaskState.COMPLETED, self.tick)
        self.events.emit(self.tick, "task_completed", robots=tuple(crew), task=t.id, late_ticks=late)
        for i in crew:
            r = self.robots[i]
            t.assigned[r.role].remove(i)
            r.task_id, r.role = None, None
            if r.status == RobotStatus.TASK:
                r.status = RobotStatus.IDLE
            elif r.status == RobotStatus.FAILED:
                r.resume_status = RobotStatus.IDLE
            r.clear_path()
        for o in self.tasks.values():                         # dependents
            if o.state == TaskState.PENDING and all(self.tasks[d].done for d in o.depends_on):
                self._make_ready(o)

    # ---- 5. charging and energy
    def _charging_and_energy(self) -> None:
        dt_h = self.config.seconds_per_tick / 3600.0
        for r in self.robots:
            if r.status == RobotStatus.STRANDED:
                continue
            e = r.spec.energy
            gain = 0.0
            if r.status == RobotStatus.CHARGE and r.charger is not None:
                c = self.chargers[r.charger]
                if r.cell == c.cell and c.occupant in (None, r.id):
                    if not r.charging:
                        c.occupant, r.charging = r.id, True
                        self.events.emit(self.tick, "charge_started", robots=(r.id,), charger=c.id, soc=round(r.soc, 4))
                    gain = min(c.power_w, e.max_charge_w) * e.charge_efficiency
            power = e.idle_w
            if r.moved:
                power += e.move_w
                if self.carrying(r):
                    t = self.tasks[r.task_id]
                    power += e.payload_w_per_kg * t.weight_kg / len(t.carriers)
            if r.worked:
                power += e.handling_w
            r.energy_used_wh += power * dt_h
            r.energy_wh = min(e.capacity_wh, r.energy_wh + (gain - power) * dt_h)
            if r.charging and r.energy_wh >= e.capacity_wh - 1e-9:
                self._stop_charging(r, "full")
                r.status = RobotStatus.IDLE
            if r.energy_wh <= 0:
                r.energy_wh = 0.0
                if r.charging or (self.config.model != "2.0" and r.charger is not None):
                    self._stop_charging(r, "empty")
                r.status = RobotStatus.STRANDED
                self.events.emit(self.tick, "robot_stranded", robots=(r.id,), task=r.task_id)
                if r.task_id is not None:
                    self.release(r.id, "stranded")
                r.charger = None

    # ───────────────────────────────────────── coalitions
    def feasible_coalitions(self, task_id: int, robots: Optional[Sequence[int]] = None, limit: int = 10,
                            per_role_candidates: int = 6, energy_reserve: float = 0.05):
        from battery_tarware.coalition import feasible_coalitions
        return feasible_coalitions(self, self.tasks[task_id], robots, limit, per_role_candidates, energy_reserve)

    # ───────────────────────────────────────── metrics
    def metrics(self) -> Dict[str, float]:
        done = [t for t in self.tasks.values() if t.state == TaskState.COMPLETED]
        lat = [t.completed_at - t.released for t in done]
        return {
            "tick": self.tick,
            "tasks_released": len(self.tasks),
            "tasks_completed": len(done),
            "tasks_open": sum(not t.done for t in self.tasks.values()),
            "demand_dropped": self.dropped_demand,
            "mean_latency": float(np.mean(lat)) if lat else None,
            "late_tasks": sum(1 for t in done if t.deadline is not None and t.completed_at > t.deadline),
            "energy_used_kwh": sum(r.energy_used_wh for r in self.robots) / 1000.0,
            "stranded": sum(r.status == RobotStatus.STRANDED for r in self.robots),
            "min_soc": min(r.soc for r in self.robots),
            "distance": sum(r.distance for r in self.robots),
        }
