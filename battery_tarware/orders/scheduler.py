"""Order scheduling: one pipeline, nine dispatching rules (+ the WSP type-weight ablation).

``OrderScheduler(rule)`` is a coordinator (``reset(env)``, ``act(env)``) for macro mode. Every tick:

1. **Charging** (shared component, identical for all rules): the greedy baseline's rule; idle robots
   below ``charge_below`` go to the nearest free compatible dock.
2. **Eligible set**: open simulator tasks with a role to fill. With the order layer these are exactly
   the released order-layer tasks (blocked ones never reach the simulator); without it, every
   released task (so the scheduler also runs on the task-only built-ins).
3. **Feasibility** (identical for all rules): ``feasible_coalitions`` over the idle robots with the
   common battery reserve (capabilities, role alternatives and work positions, payload, reachability,
   energy). An eligible task without a feasible coalition gets a rejection reason:
   ``infeasible:capability`` (no operational robot can fill a role), ``infeasible:payload``,
   ``unavailable:robots_busy`` / ``unavailable:charging`` / ``unavailable:breakdown`` (capable robots
   exist but are mostly committed to tasks / charging or heading to docks / broken down),
   ``infeasible:battery`` (feasible only without the reserve), ``infeasible:unreachable``.
4. **Assignment**: the rule ranks the feasible (task, coalition) pairs; pairs are committed in rank
   order when their robots are still free, atomically, up to ``max_assign_per_tick``; tasks whose
   coalitions all lost a robot get one re-query on the remaining robots. Task rules rank tasks and
   take the coalition with the earliest estimated completion (the greedy's crew choice), so they
   differ only in sequencing; pair rules (NSP, ZLP_2D, PLB) also choose the crew.
5. **Decision log**: one ``schedule_decision`` event per tick with eligible tasks: per task its score,
   number of feasible coalitions, best estimated completion, and outcome (``assigned``,
   ``robot_taken``, ``budget``, or the rejection reason), plus the committed crews and the number of
   candidate pairs evaluated. Wall-clock compute time is kept in ``scheduler.runtime`` (not in the
   event log, which stays deterministic).

**Budgets.** The *algorithmic* budget is count-based and deterministic: at most
``max_assign_per_tick`` commits, ``coalitions_per_task`` coalitions per task and
``per_role_candidates`` robots per role, one re-query. It bounds the work per tick independently
of the machine. ``deadline_ms`` (default 20 ms) is a *wall-clock* evaluation target only: each tick's
measured compute time (charging rule + assignment, ``time.perf_counter``) is compared with it and
violations are reported (``metrics.scheduling.deadline_*``). The scheduler never stops early
because of it, so decisions do not depend on machine speed or load.

All rules see the same candidate set and the same information (current and past state only).
Time terms use simulation ticks. Ties break by order arrival, order-layer task id, then estimated
completion, then robot ids.

Rule keys (``norm`` = min-max over the current candidate set, 0 if all equal):

=========  =====  ======================================================================
fifo       task   smallest arrival a_j (order arrival; release tick without the order layer)
edd        task   smallest due date d_j (none last)
lwf        task   largest waiting time since the task (or one of its roles) last became open
wsp        task   largest 0.45·norm(priority) + 0.30·min(wait/100, 1) + 0.25·T(op);
                  T: outbound 0.8, inbound 0.6, relocation 0.4
wsp_eq     task   WSP with T = 0.6 for every operation (ablation: sequencing vs type favouritism)
spt        task   smallest remaining processing estimate p_j (stage work + carried legs)
cr         task   smallest (d_j − t) / max(p_j, 1) (no due date last)
nsp        pair   smallest crew arrival estimate (largest member travel time to its first work cell)
zlp_2d     pair   largest 0.8·A + 0.2·norm(priority); A = 1.0 / 0.7 / 0.3 when both / one / neither
                  of the task's first work cell and destination lie in the carrier's current zone
plb        pair   largest 0.4·(1 − C) + 0.3·B + 0.3·(1 − L): L = norm(crew path length: members'
                  legs to their first work cell + carried legs); C = share of the carrier's planned
                  cells (A* to the payload and between stage cells) on cells or planned paths of
                  other committed robots; B = 1 − norm(crew mean busy share over the last 100 ticks)
=========  =====  ======================================================================

Zones (ZLP_2D): by default the rack blocks (4-connected groups of storage cells), each map cell
belonging to the nearest block (BFS; ties to the lower block id); ``zones`` may give an explicit
(H, W) integer array.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from battery_tarware.coalition import _dist
from battery_tarware.coordination import GreedyCoalitionCoordinator, _processing_estimate
from battery_tarware.navigation import MOVES, astar
from battery_tarware.robots import RobotStatus

TASK_RULES = ("fifo", "edd", "lwf", "wsp", "wsp_eq", "spt", "cr")
PAIR_RULES = ("nsp", "zlp_2d", "plb")
RULE_NAMES = TASK_RULES + PAIR_RULES
TYPE_SCORES = {"outbound": 0.8, "inbound": 0.6, "relocation": 0.4}


@dataclass
class TaskInfo:
    arrival: int
    priority: float
    due: Optional[int]
    op: str
    ready_since: int
    order_task: Optional[str]          # order-layer task id (None without the order layer)
    order_id: Optional[str]


def task_info(sim, t) -> TaskInfo:
    m = getattr(sim, "orders", None)
    ot = m.task_for_sim(t.id) if m is not None else None
    if ot is None:
        return TaskInfo(t.released, float(t.type.priority), t.deadline, "outbound", t.released, None, None)
    o = m.orders[ot.order_id]
    since = ot.ready_since if ot.ready_since is not None else t.released
    return TaskInfo(o.arrival, o.priority, o.due, ot.op, since, ot.task_id, o.order_id)


def rack_zones(sim) -> np.ndarray:
    """Zone id per cell: 4-connected storage blocks, every cell assigned to its nearest block."""
    H, W = sim.layout.shape
    zone = -np.ones((H, W), dtype=np.int32)
    storage = set(sim.layout.cells("storage"))
    nid = 0
    for c in sorted(storage):
        if zone[c] >= 0:
            continue
        stack = [c]
        zone[c] = nid
        while stack:
            y, x = stack.pop()
            for dy, dx in MOVES:
                n = (y + dy, x + dx)
                if n in storage and zone[n] < 0:
                    zone[n] = nid
                    stack.append(n)
        nid += 1
    frontier = deque(sorted(storage, key=lambda c: (zone[c], c)))
    while frontier:                               # multi-source BFS over the whole grid (walls too)
        y, x = frontier.popleft()
        for dy, dx in MOVES:
            n = (y + dy, x + dx)
            if 0 <= n[0] < H and 0 <= n[1] < W and zone[n] < 0:
                zone[n] = zone[y, x]
                frontier.append(n)
    return zone


class OrderScheduler:
    def __init__(self, rule: str = "fifo", charge_below: float = 0.3, energy_reserve: float = 0.15,
                 max_assign_per_tick: int = 8, coalitions_per_task: int = 4, per_role_candidates: int = 6,
                 wsp_weights: Tuple[float, float, float] = (0.45, 0.30, 0.25),
                 type_scores: Optional[Dict[str, float]] = None, wait_norm: int = 100,
                 zlp_weights: Tuple[float, float] = (0.8, 0.2), zone_affinity: Tuple[float, float, float] = (1.0, 0.7, 0.3),
                 plb_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3), busy_window: int = 100,
                 zones: Optional[np.ndarray] = None, log_decisions: bool = True, deadline_ms: float = 20.0):
        if rule not in RULE_NAMES:
            raise ValueError("unknown rule %r; known: %s" % (rule, RULE_NAMES))
        self.rule = rule
        self.charge_below, self.energy_reserve = charge_below, energy_reserve
        self.max_assign = max_assign_per_tick
        self.k, self.per_role = coalitions_per_task, per_role_candidates
        self.wsp_w, self.wait_norm = wsp_weights, wait_norm
        self.type_scores = dict(type_scores or TYPE_SCORES)
        if rule == "wsp_eq":
            self.type_scores = {k: 0.6 for k in self.type_scores}
        self.zlp_w, self.affinity = zlp_weights, zone_affinity
        self.plb_w, self.busy_window = plb_weights, busy_window
        self._zones_given = zones
        self.log_decisions = log_decisions
        self._charger = GreedyCoalitionCoordinator("fifo", charge_below=charge_below, energy_reserve=energy_reserve)
        self.deadline_ms = float(deadline_ms)                    # wall-clock target, reported only
        # per tick: (tick, seconds, candidate pairs, eligible tasks, free robots, charging-rule seconds)
        self.runtime: List[Tuple[int, float, int, int, int, float]] = []
        self._env_id = None

    # ---------------------------------------------------------------- coordinator API
    def reset(self, env) -> None:
        sim = env.sim if hasattr(env, "sim") else env.unwrapped.sim if hasattr(env, "unwrapped") else env
        self.runtime = []
        self._busy: Dict[int, deque] = {r.id: deque(maxlen=self.busy_window) for r in sim.robots}
        self._zones = self._zones_given if self._zones_given is not None else rack_zones(sim)
        self._env_id = (id(sim), sim.seed)

    def act(self, env) -> np.ndarray:
        sim = env.sim if hasattr(env, "sim") else env.unwrapped.sim if hasattr(env, "unwrapped") else env
        if sim.config.action_mode != "macro":
            raise ValueError("OrderScheduler needs action_mode='macro'")
        if self._env_id != (id(sim), sim.seed) or sim.tick == 0:
            self.reset(sim)
        for r in sim.robots:
            self._busy[r.id].append(r.status in (RobotStatus.TASK, RobotStatus.CHARGE))
        t0 = time.perf_counter()
        self._charger._charging(sim)                                   # 1. shared charging rule
        t1 = time.perf_counter()
        eligible = [t for t in sim.open_tasks() if t.joinable_roles()]   # 2. eligible set
        n_pairs, n_free = 0, 0
        if eligible:
            n_pairs, n_free = self._assign(sim, eligible)
        self.runtime.append((sim.tick, time.perf_counter() - t0, n_pairs, len(eligible), n_free, t1 - t0))
        return np.zeros(len(sim.robots), dtype=np.int64)

    # ---------------------------------------------------------------- pipeline
    def _free(self, sim) -> List[int]:
        return [r.id for r in sim.robots if r.status == RobotStatus.IDLE and r.operational]

    def _feasible(self, sim, t, free):
        return sim.feasible_coalitions(t.id, robots=free, limit=self.k, per_role_candidates=self.per_role,
                                       energy_reserve=self.energy_reserve)

    def _reject_reason(self, sim, t, free) -> str:
        roles = t.joinable_roles()
        cr = t.type.carrier_role

        def fits(r, role):
            return t.type.role(role).accepts(r) and (role != cr or r.spec.payload_kg > 0)

        ops = [r for r in sim.robots if r.operational]
        if not _staffable(ops, roles, fits):
            return "infeasible:capability"
        if cr in roles:
            have = sum(sim.robots[i].spec.payload_kg for i in t.assigned.get(cr, []))
            best = sorted((r.spec.payload_kg for r in ops if fits(r, cr)), reverse=True)
            if have + sum(best[:roles[cr]]) + 1e-9 < t.weight_kg:
                return "infeasible:payload"
        fr = [sim.robots[i] for i in free]
        if cr in roles:                      # the carriers strong enough for the load are all busy
            have = sum(sim.robots[i].spec.payload_kg for i in t.assigned.get(cr, []))
            best_free = sorted((r.spec.payload_kg for r in fr if fits(r, cr)), reverse=True)
            if have + sum(best_free[:roles[cr]]) + 1e-9 < t.weight_kg:
                fr = [r for r in fr if not fits(r, cr)]          # report why the heavy carriers are away
        if not _staffable(fr, roles, fits):
            busy = [r for r in sim.robots if r.id not in free and any(fits(r, role) for role in roles)]
            charging = sum(r.status == RobotStatus.CHARGE for r in busy)
            broken = sum(not r.operational for r in busy)
            tasked = sum(r.status == RobotStatus.TASK for r in busy)
            if broken > max(charging, tasked):
                return "unavailable:breakdown"
            return "unavailable:charging" if charging > tasked else "unavailable:robots_busy"
        if sim.feasible_coalitions(t.id, robots=free, limit=1, per_role_candidates=self.per_role,
                                   energy_reserve=-1e9):
            return "infeasible:battery"
        return "infeasible:unreachable"

    def _assign(self, sim, eligible) -> int:
        free = self._free(sim)
        infos = {t.id: task_info(sim, t) for t in eligible}
        status: Dict[int, str] = {}
        ncoal: Dict[int, int] = {}
        best_est: Dict[int, float] = {}
        pairs = []
        for t in eligible:
            cs = self._feasible(sim, t, free) if free else []
            ncoal[t.id] = len(cs)
            if not cs:
                status[t.id] = self._reject_reason(sim, t, free)
                continue
            best_est[t.id] = cs[0].est_ticks
            pairs += [(t, c) for c in cs]
        scores = self._scores(sim, pairs, infos) if pairs else []
        order = sorted(range(len(pairs)), key=lambda i: scores[i])
        taken: set = set()
        done: Dict[int, Tuple[int, ...]] = {}
        n_eval = len(pairs)

        def commit(t, c) -> bool:
            if len(done) >= self.max_assign or t.id in done or any(r in taken for r in c.robots):
                return False
            ok, _ = sim.assign_coalition(t.id, {k: list(v) for k, v in c.assignment.items()})
            if ok:
                done[t.id] = c.robots
                taken.update(c.robots)
            return ok

        for i in order:
            commit(*pairs[i])
        # one re-query for tasks whose coalitions all lost a robot to a higher-ranked task
        retry = [t for t in eligible if ncoal[t.id] and t.id not in done]
        if retry and len(done) < self.max_assign:
            rest = [i for i in self._free(sim)]
            pairs2 = []
            for t in retry:
                pairs2 += [(t, c) for c in (self._feasible(sim, t, rest) if rest else [])]
            n_eval += len(pairs2)
            if pairs2:
                sc2 = self._scores(sim, pairs2, infos)
                for i in sorted(range(len(pairs2)), key=lambda i: sc2[i]):
                    commit(*pairs2[i])
        for t in eligible:
            if t.id in done:
                status[t.id] = "assigned"
            elif t.id not in status:
                status[t.id] = "budget" if len(done) >= self.max_assign else "robot_taken"
        if self.log_decisions:
            pair_score = {}
            for i, (t, c) in enumerate(pairs):
                if t.id not in pair_score:
                    pair_score[t.id] = scores[i]
            sim.events.emit(sim.tick, "schedule_decision", (), None, rule=self.rule, free=len(free),
                            pairs=n_eval, assigned={str(k): list(v) for k, v in done.items()},
                            tasks=[[t.id, infos[t.id].order_task, _round(pair_score.get(t.id)), ncoal[t.id],
                                    _round(best_est.get(t.id)), status[t.id]] for t in eligible])
        return n_eval, len(free)

    # ---------------------------------------------------------------- scores (smaller sorts first)
    def _scores(self, sim, pairs, infos) -> List[Tuple]:
        rule, now = self.rule, sim.tick
        tids = sorted({t.id for t, _ in pairs})
        pr = _norm({i: infos[i].priority for i in tids})

        def tail(t, c):
            inf = infos[t.id]
            return (inf.arrival, inf.order_task or "", t.id, c.est_ticks, c.robots)

        out = []
        if rule in TASK_RULES:
            for t, c in pairs:
                inf = infos[t.id]
                if rule == "fifo":
                    k = (inf.arrival,)
                elif rule == "edd":
                    k = (math.inf if inf.due is None else inf.due,)
                elif rule == "lwf":
                    k = (-(now - inf.ready_since),)
                elif rule in ("wsp", "wsp_eq"):
                    a, b, g = self.wsp_w
                    s = a * pr[t.id] + b * min(max(now - inf.ready_since, 0) / self.wait_norm, 1.0) + \
                        g * self.type_scores.get(inf.op, 0.0)
                    k = (-s,)
                elif rule == "spt":
                    k = (_processing_estimate(sim, t),)
                else:
                    k = (math.inf if inf.due is None else (inf.due - now) / max(_processing_estimate(sim, t), 1.0),)
                out.append(k + tail(t, c))
            return out
        if rule == "nsp":
            return [(self._crew_arrival(sim, t, c),) + tail(t, c) for t, c in pairs]
        if rule == "zlp_2d":
            wa, wp = self.zlp_w
            return [(-(wa * self._affinity(sim, t, c) + wp * pr[t.id]),) + tail(t, c) for t, c in pairs]
        # plb
        L = _norm({i: self._path_length(sim, t, c) for i, (t, c) in enumerate(pairs)})
        busy = _norm({i: np.mean([np.mean(self._busy[r]) if self._busy[r] else 0.0 for r in c.robots])
                      for i, (t, c) in enumerate(pairs)})
        occ = self._occupied(sim)
        wc, wb, wl = self.plb_w
        for i, (t, c) in enumerate(pairs):
            C = self._conflict(sim, t, c, occ)
            s = wc * (1 - C) + wb * (1 - busy[i]) + wl * (1 - L[i])
            out.append((-s,) + tail(t, c))
        return out

    # ---------------------------------------------------------------- components
    @staticmethod
    def _first_cell(t, role):
        if role == t.type.carrier_role and not t.carried:
            return t.payload_cell
        s = t.next_stage_for(role)
        return t.stage_cells[s if s is not None else t.stage]

    def _member_legs(self, sim, t, c):
        for role, ids in c.assignment.items():
            for i in ids:
                r = sim.robots[i]
                cell = self._first_cell(t, role)
                d = _dist(sim, r, r.cell, cell, False)
                if not np.isfinite(d):             # adjacent work position: the cell itself may be closed
                    d = min((_dist(sim, r, r.cell, (cell[0] + dy, cell[1] + dx), False)
                             for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                             if sim.layout.in_bounds((cell[0] + dy, cell[1] + dx))), default=math.inf)
                yield r, role, d

    def _crew_arrival(self, sim, t, c) -> float:
        return max((d / r.spec.speed for r, _, d in self._member_legs(sim, t, c)), default=0.0)

    def _carrier(self, sim, t, c):
        cr = t.type.carrier_role
        ids = c.assignment.get(cr) or t.assigned.get(cr) or c.robots
        return sim.robots[min(ids)]

    def _affinity(self, sim, t, c) -> float:
        z = self._zones
        home = z[self._carrier(sim, t, c).cell]
        first = t.payload_cell if (t.type.carrier_role and not t.carried) else t.stage_cell
        n = int(z[first] == home) + int(z[t.destination] == home)
        return self.affinity[2 - n]

    def _carried_legs(self, sim, t, carrier) -> float:
        cells = list(t.stage_cells[t.stage:])
        return sum(_dist(sim, carrier, a, b, True) for a, b in zip(cells, cells[1:]) if a != b)

    def _path_length(self, sim, t, c) -> float:
        travel = sum(d for _, _, d in self._member_legs(sim, t, c))
        return travel + self._carried_legs(sim, t, self._carrier(sim, t, c))

    @staticmethod
    def _occupied(sim) -> set:
        occ = set()
        for r in sim.robots:
            if r.status in (RobotStatus.TASK, RobotStatus.CHARGE) or not r.operational:
                occ.add(r.cell)
                occ.update(r.path)
        return occ

    def _conflict(self, sim, t, c, occ) -> float:
        r = self._carrier(sim, t, c)
        cache = sim.__dict__.setdefault("_plb_path_cache", {})
        if cache.get("tick") != sim.tick:
            cache.clear()
            cache["tick"] = sim.tick
        cells: List = []
        start = r.cell
        legs = []
        if not t.carried:
            legs.append((start, t.payload_cell, False))
            start = t.payload_cell
        for a in t.stage_cells[t.stage:]:
            if a != start:
                legs.append((start, a, True))
                start = a
        for a, b, loaded in legs:
            key = (a, b, tuple(sorted(r.zones(loaded))))
            if key not in cache:
                cache[key] = astar(sim.passable(key[2]), a, b, None, sim.layout.directions) or []
            cells += cache[key]
        others = occ - {sim.robots[i].cell for i in c.robots}
        return sum(1 for x in cells if x in others) / max(1, len(cells))


def _staffable(robots, roles: Dict[str, int], fits) -> bool:
    """Can distinct robots fill every open slot (bipartite matching, Kuhn's algorithm)?"""
    slots = [role for role, n in sorted(roles.items()) for _ in range(n)]
    match: Dict[int, int] = {}                     # robot index -> slot index

    def augment(j, seen) -> bool:
        for i, r in enumerate(robots):
            if i in seen or not fits(r, slots[j]):
                continue
            seen.add(i)
            if i not in match or augment(match[i], seen):
                match[i] = j
                return True
        return False
    return all(augment(j, set()) for j in range(len(slots)))


def _norm(d: Dict) -> Dict:
    if not d:
        return {}
    v = np.array(list(d.values()), dtype=float)
    finite = v[np.isfinite(v)]
    lo, hi = (finite.min(), finite.max()) if len(finite) else (0.0, 0.0)
    out = {}
    for k, x in d.items():
        if not np.isfinite(x):
            out[k] = 1.0
        else:
            out[k] = 0.0 if hi - lo < 1e-12 else (x - lo) / (hi - lo)
    return out


def _round(x):
    if x is None:
        return None
    if isinstance(x, tuple):
        x = x[0]
    return None if not np.isfinite(x) else round(float(x), 4)
