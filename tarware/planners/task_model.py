"""Shared task model for the centralized planner baselines (BMRTA, Calvo).

Converts the warehouse request queue into role-typed coalition tasks with
travel/handling duration estimates and battery-feasibility checks. Battery
accounting mirrors the environment's actual step costs (``tarware.warehouse``)
rather than the abstract ``EnergyModel`` planning units, so recharge insertion
operates on the same scale as the executed battery dynamics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tarware.definitions import PACKAGE_REWARD_SCALE, AgentType
from tarware.heuristic import (
    AssignmentContext,
    _nearest_charger_path_len,
    _path_len,
)

# Mirrors tarware.warehouse battery constants.
MOVE_COST = 100.0 / 8640.0      # _BATTERY_CONSUMPTION_MOVE (unloaded, per step)
LOAD_COST = 0.15                # _BATTERY_CONSUMPTION_LOAD / _PICKER_LIFT_ENERGY
LOADED_FACTOR_MAX = 1.5         # worst-case payload multiplier (7 kg)
HANDLE_STEPS = 2                # TOGGLE_LOAD + cross-role handoff synchronisation

# `_path_len` returns 1e6 when no path exists; anything close is unreachable.
UNREACHABLE = int(1e5)


@dataclass
class Task:
    """One outbound request expressed as a coalition task."""

    shelf: object
    shelf_id: int
    package_type: object
    loc_id: int
    yx: Tuple[int, int]
    created_step: int
    n_agvs: int
    n_pickers: int
    reward_scale: float

    @property
    def coalition_size(self) -> int:
        return self.n_agvs + self.n_pickers


class BasePlanner:
    """Interface consumed by ``heuristic_episode(assignment_fn=...)``."""

    name: str = "base"

    def replan(self, env, ctx: AssignmentContext) -> list:
        raise NotImplementedError

    def on_event(self, event: str, **kw) -> None:  # pragma: no cover - default no-op
        pass

    def stats(self) -> dict:
        return {}


def update_request_ages(ctx: AssignmentContext, created_step_by_shelf: Dict[int, int]) -> None:
    """Track first-seen step per queued shelf; forget shelves that left the queue.

    A shelf re-entering the queue later (after an inbound/return cycle) restarts
    its age, which is the intended semantics for request urgency.
    """
    queue_ids = {int(item.id) for item in ctx.request_queue}
    for stale in [sid for sid in created_step_by_shelf if sid not in queue_ids]:
        del created_step_by_shelf[stale]
    for item in ctx.request_queue:
        created_step_by_shelf.setdefault(int(item.id), ctx.timestep)


def build_tasks(
    env,
    ctx: AssignmentContext,
    created_step_by_shelf: Optional[Dict[int, int]] = None,
) -> List[Task]:
    """Unassigned queue requests as ``Task`` records (assigned/carried skipped)."""
    assigned_shelf_ids = set(ctx.assigned_items.values())
    tasks: List[Task] = []
    for item in ctx.request_queue:
        loc_id = ctx.location_id(item.y, item.x)
        if loc_id is None:
            continue
        if item.id in assigned_shelf_ids or item.id in ctx.carried_shelf_ids:
            continue
        created = ctx.timestep
        if created_step_by_shelf is not None:
            created = created_step_by_shelf.get(int(item.id), ctx.timestep)
        tasks.append(
            Task(
                shelf=item,
                shelf_id=int(item.id),
                package_type=item.package_type,
                loc_id=int(loc_id),
                yx=(int(item.y), int(item.x)),
                created_step=int(created),
                n_agvs=int(getattr(item, "required_agvs", 1)),
                n_pickers=int(getattr(item, "required_pickers", 1)),
                reward_scale=float(PACKAGE_REWARD_SCALE.get(item.package_type, 1.0)),
            )
        )
    return tasks


def cached_travel(env, agent, yx: Tuple[int, int], cache: Optional[dict]) -> int:
    """A* travel steps with an optional per-replan cache keyed by (agent, target)."""
    if cache is None:
        return _path_len(env, agent, yx)
    key = (int(agent.id), yx)
    if key not in cache:
        cache[key] = _path_len(env, agent, yx)
    return cache[key]


def goal_leg_steps(env, task: Task) -> int:
    """Manhattan estimate of the loaded shelf-to-nearest-goal leg."""
    y, x = task.yx
    return min(abs(x - gx) + abs(y - gy) for gx, gy in env.goals)


def coalition_completion_steps(
    env,
    task: Task,
    agvs: list,
    pickers: list,
    cache: Optional[dict] = None,
) -> int:
    """Estimated completion: latest member arrival + handling + delivery leg.

    The max-arrival term encodes the cross-role synchronisation (the AGV can
    only load once the required Pickers are at the shelf).
    """
    arrivals = [cached_travel(env, a, task.yx, cache) for a in list(agvs) + list(pickers)]
    return (max(arrivals) if arrivals else 0) + HANDLE_STEPS + goal_leg_steps(env, task)


def battery_feasible(
    env,
    agent,
    task: Task,
    margin: float,
    cache: Optional[dict] = None,
) -> bool:
    """Can `agent` serve `task` and still reach a charger above `margin` battery?

    Uses the environment's real battery scale with a worst-case payload
    multiplier on the loaded leg.
    """
    to_shelf = cached_travel(env, agent, task.yx, cache)
    if to_shelf >= UNREACHABLE:
        return False
    carry = goal_leg_steps(env, task)
    if agent.type == AgentType.AGV:
        cost = (
            to_shelf * MOVE_COST
            + carry * MOVE_COST * LOADED_FACTOR_MAX
            + 2.0 * LOAD_COST
        )
    else:
        # PICKER_SOLO: the picker itself carries the item to the goal.
        picker_carries = task.n_agvs == 0
        cost = to_shelf * MOVE_COST + LOAD_COST
        if picker_carries:
            cost += carry * MOVE_COST + LOAD_COST
    reserve = _nearest_charger_path_len(env, agent) * MOVE_COST
    return float(agent.battery) - cost - reserve >= float(margin)
