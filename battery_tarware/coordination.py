"""Coordinator plug-in interface and baseline coordinators.

A coordinator looks at the environment and returns one action per robot (``act``); centralized
coordinators may also commit crews directly through the simulator API (``assign_coalition``,
``send_to_charger``, ``release``), which the environment validates exactly like actions.

Dispatching rules (``RULES``) only order tasks; crew selection is separate (``feasible_coalitions``),
because a priority rule cannot decide which robots can serve a task together.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np

from battery_tarware.robots import RobotStatus


class Coordinator(Protocol):
    def reset(self, env) -> None: ...
    def act(self, env) -> np.ndarray: ...


def _processing_estimate(sim, t) -> float:
    """Stage work plus the carried distance between stage cells (ticks, unit speed)."""
    work = sum(s.duration for s in t.type.stages[t.stage:])
    cells = list(t.stage_cells[t.stage:])
    travel = sum(sim.manhattan(a, b) for a, b in zip(cells, cells[1:]))
    return float(work + travel)


@dataclass(frozen=True)
class Rule:
    name: str
    full_name: str
    definition: str
    key: Callable                        # (sim, task) -> sort key, smallest first


RULES: Dict[str, Rule] = {r.name: r for r in [
    Rule("fifo", "First In, First Out",
         "serve tasks in release order: key = (release tick, task id)",
         lambda sim, t: (t.released, t.id)),
    Rule("edd", "Earliest Due Date",
         "smallest absolute deadline first; tasks without deadline last: key = (deadline or +inf, release, id)",
         lambda sim, t: (math.inf if t.deadline is None else t.deadline, t.released, t.id)),
    Rule("spt", "Shortest Processing Time",
         "smallest remaining processing estimate p = sum of remaining stage durations + Manhattan length of "
         "the remaining stage-to-stage route: key = (p, release, id)",
         lambda sim, t: (_processing_estimate(sim, t), t.released, t.id)),
    Rule("cr", "Critical Ratio",
         "smallest (deadline - now) / p first, p as in SPT; tasks without deadline last",
         lambda sim, t: ((math.inf if t.deadline is None else (t.deadline - sim.tick) / max(1.0, _processing_estimate(sim, t))),
                         t.released, t.id)),
    Rule("priority", "Highest Priority First",
         "largest task-type priority first, ties FIFO: key = (-priority, release, id)",
         lambda sim, t: (-t.type.priority, t.released, t.id)),
]}


class RandomCoordinator:
    """Uniformly random valid action per robot (masked); a sanity baseline for RL comparisons."""

    def __init__(self, seed: Optional[int] = None, noop_prob: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.noop_prob = noop_prob

    def reset(self, env) -> None:
        pass

    def act(self, env) -> np.ndarray:
        masks = env.action_masks()
        out = np.zeros(len(masks), dtype=np.int64)
        for i, m in enumerate(masks):
            valid = np.flatnonzero(m[1:]) + 1
            if len(valid) and self.rng.random() >= self.noop_prob:
                out[i] = int(self.rng.choice(valid))
        return out


class GreedyCoalitionCoordinator:
    """Centralized baseline for macro mode.

    Each tick: (1) idle robots below ``charge_below`` state of charge go to the nearest free
    compatible charger; (2) open tasks, ordered by ``rule``, get the best feasible coalition of
    idle robots (fastest estimated completion, then least energy), committed atomically. Robots
    keep a battery reserve (``energy_reserve``) when coalitions are checked for feasibility.
    """

    def __init__(self, rule: str = "fifo", charge_below: float = 0.3, energy_reserve: float = 0.15,
                 max_tasks_per_tick: int = 8):
        if rule not in RULES:
            raise ValueError("unknown rule %r; known: %s" % (rule, sorted(RULES)))
        self.rule = RULES[rule]
        self.charge_below = charge_below
        self.energy_reserve = energy_reserve
        self.max_tasks_per_tick = max_tasks_per_tick

    def reset(self, env) -> None:
        pass

    def act(self, env) -> np.ndarray:
        sim = env.sim if hasattr(env, "sim") else env
        if sim.config.action_mode != "macro":
            raise ValueError("GreedyCoalitionCoordinator needs action_mode='macro'")
        self._charging(sim)
        tasks = sorted((t for t in sim.open_tasks() if t.joinable_roles()), key=lambda t: self.rule.key(sim, t))
        n = 0
        for t in tasks:
            if n >= self.max_tasks_per_tick:
                break
            options = sim.feasible_coalitions(t.id, limit=1, energy_reserve=self.energy_reserve)
            if options and sim.assign_coalition(t.id, {k: list(v) for k, v in options[0].assignment.items()})[0]:
                n += 1
        return np.zeros(len(sim.robots), dtype=np.int64)

    def _charging(self, sim) -> None:
        targeted = {r.charger for r in sim.robots if r.charger is not None}
        for r in sim.robots:
            if r.status != RobotStatus.IDLE or not r.operational or r.soc >= self.charge_below:
                continue
            options = [c for c in sim.chargers if c.id not in targeted and c.occupant is None
                       and sim.can_charge(r, c)[0]]
            if not options:
                continue
            c = min(options, key=lambda c: (sim.manhattan(r.cell, c.cell), -c.power_w, c.id))
            if sim.send_to_charger(r.id, c.id)[0]:
                targeted.add(c.id)


def run_episode(env, coordinator, seed: Optional[int] = None, max_steps: Optional[int] = None) -> Dict[str, float]:
    """Roll out one episode and return the final metrics plus the total team reward."""
    obs, info = env.reset(seed=seed)
    coordinator.reset(env)
    total = 0.0
    steps = 0
    while True:
        obs, reward, term, trunc, info = env.step(coordinator.act(env))
        total += reward
        steps += 1
        if term or trunc or (max_steps is not None and steps >= max_steps):
            break
    m = dict(info["metrics"])
    m["return"] = total
    return m
