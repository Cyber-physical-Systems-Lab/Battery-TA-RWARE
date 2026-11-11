"""Feasible coalitions: task requirements + robot capabilities + joint execution constraints.

The environment decides *which teams can do a task*; choosing among them is the coordinator's job.
A coalition is feasible when

* every open role is filled by distinct, operational, uncommitted robots with the role's capability;
* the carrier role's combined payload capacity covers the task weight (joint transport adds up);
* each member can reach the cells it must visit within its allowed zones; and
* each member's estimated energy for the task (travel to its cells, the carried legs and the stage
  work it takes part in, plus idle draw) leaves at least ``energy_reserve`` of its battery.

Estimates use grid distances (BFS within the robot's zones), robot speed and the robot's power
model, so they are optimistic about congestion. Results are deterministic and sorted by estimated
completion time, then total energy.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from battery_tarware.navigation import bfs_distances
from battery_tarware.robots import Robot, RobotStatus


@dataclass(frozen=True)
class Coalition:
    task_id: int
    assignment: Dict[str, Tuple[int, ...]]       # role -> robot ids (open roles only)
    est_ticks: float                              # estimated ticks until the task completes
    est_energy_wh: Dict[int, float]               # per member
    payload_capacity_kg: float

    @property
    def robots(self) -> Tuple[int, ...]:
        return tuple(i for ids in self.assignment.values() for i in ids)


def _dist(sim, r: Robot, a, b, carrying: bool) -> float:
    # distances on the 4-connected grid are symmetric: search from the (task) cell b, so the cache is
    # bounded by the number of task cells rather than by every position a robot ever had
    key = (tuple(sorted(r.zones(carrying))), b)
    cache = sim.__dict__.setdefault("_bfs_cache", {})
    if key not in cache:
        if len(cache) > 4096:
            cache.clear()
        cache[key] = bfs_distances(sim.passable(key[0]), b)
    d = cache[key][a]
    return float("inf") if d < 0 else float(d)


def estimate(sim, t, robot: Robot, role: str, start=None) -> Tuple[float, float]:
    """(ticks until this robot finishes its part, energy in Wh) for ``robot`` in ``role`` of ``t``."""
    e = robot.spec.energy
    dt_h = sim.config.seconds_per_tick / 3600.0
    cell = start or robot.cell
    carrier = role == t.type.carrier_role
    ticks = 0.0
    energy = 0.0
    carrying = t.carried and carrier
    if carrier and not t.carried:
        d = _dist(sim, robot, cell, t.payload_cell, False)
        ticks += d / robot.spec.speed
        energy += d / robot.spec.speed * e.move_w * dt_h
        cell = t.payload_cell
    for i in range(t.stage, len(t.type.stages)):
        st = t.type.stages[i]
        target = t.stage_cells[i]
        if role in st.roles:
            if cell != target:
                d = _dist(sim, robot, cell, target, carrying)
                rs = t.type.role(role)
                opt = rs.option_for(robot)
                if rs.alternatives and opt is not None and opt.presence == "adjacent" and not carrier:
                    # a role alternative worked from beside the cell: aim at the nearest cell within radius
                    rad = opt.radius
                    d = min((_dist(sim, robot, cell, (target[0] + dy, target[1] + dx), False)
                             for dy in range(-rad, rad + 1) for dx in range(-rad, rad + 1)
                             if (dy or dx) and sim.layout.in_bounds((target[0] + dy, target[1] + dx))), default=d)
                elif d == float("inf") and not carrier:   # adjacent presence: aim next to the cell
                    d = min((_dist(sim, robot, cell, (target[0] + dy, target[1] + dx), False)
                             for dy in (-1, 0, 1) for dx in (-1, 0, 1)), default=float("inf"))
                move_t = d / robot.spec.speed
                ticks += move_t
                share = t.weight_kg / max(1, t.type.role(t.type.carrier_role).count) if carrying else 0.0
                energy += move_t * (e.move_w + e.payload_w_per_kg * share) * dt_h
                cell = target
            work = st.duration / robot.spec.handling_speed
            ticks += work
            energy += work * e.handling_w * dt_h
            if carrier and i == 0:
                carrying = True
    energy += ticks * e.idle_w * dt_h
    return ticks, energy


def feasible_coalitions(sim, t, robots: Optional[Sequence[int]] = None, limit: int = 10,
                        per_role_candidates: int = 6, energy_reserve: float = 0.05) -> List[Coalition]:
    if t.done:
        return []
    pool = [sim.robots[i] for i in (robots if robots is not None else range(len(sim.robots)))]
    pool = [r for r in pool if r.operational and r.status == RobotStatus.IDLE]
    open_roles = t.joinable_roles()
    if not open_roles:
        return []
    per_role: Dict[str, List[Tuple[Robot, float, float]]] = {}
    for role, n in open_roles.items():
        rs = t.type.role(role)
        cands = []
        for r in pool:
            if not rs.accepts(r):
                continue
            if role == t.type.carrier_role and r.spec.payload_kg <= 0:
                continue
            ticks, wh = estimate(sim, t, r, role)
            if not np.isfinite(ticks):
                continue
            if r.energy_wh - wh < energy_reserve * r.spec.energy.capacity_wh:
                continue
            cands.append((r, ticks, wh))
        cands.sort(key=lambda c: (c[1], c[0].id))
        if len(cands) < n:
            return []
        per_role[role] = cands[:max(n, per_role_candidates)]
    roles = list(open_roles)
    have_cap = sum(sim.robots[i].spec.payload_kg for i in t.assigned.get(t.type.carrier_role, [])) \
        if t.type.carrier_role else 0.0
    out: List[Coalition] = []
    combos = [itertools.combinations(per_role[role], open_roles[role]) for role in roles]
    for pick in itertools.product(*combos):
        ids = [c[0].id for group in pick for c in group]
        if len(set(ids)) != len(ids):
            continue
        cap = have_cap
        if t.type.carrier_role in open_roles:
            cap += sum(c[0].spec.payload_kg for c in pick[roles.index(t.type.carrier_role)])
        if t.type.carrier_role and cap + 1e-9 < t.weight_kg:
            continue
        ticks = max(c[1] for group in pick for c in group)
        energy = {c[0].id: c[2] for group in pick for c in group}
        out.append(Coalition(t.id, {role: tuple(c[0].id for c in group) for role, group in zip(roles, pick)},
                             ticks, energy, cap))
    out.sort(key=lambda c: (c.est_ticks, sum(c.est_energy_wh.values()), c.robots))
    return out[:limit]
