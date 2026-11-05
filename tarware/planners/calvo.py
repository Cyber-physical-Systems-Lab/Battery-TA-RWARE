"""Calvo & Capitán-style heterogeneous coalition planner with plan repair.

Reimplementation of the main heuristic ("version 9") of Calvo & Capitán,
"Optimal Task Allocation for Heterogeneous Multi-robot Teams with Battery
Constraints" (ICRA 2024; T-RO 2025; repo mrta_heuristic_planner), adapted
from its offline MATLAB form to a receding-horizon planner:

* Allocation order: tasks are ranked by a reward coefficient (package reward
  scale + an urgency term on request age) and inserted greedily, matching the
  reward-coefficient ordering that distinguishes version 9 from the random /
  pseudo-random variants of the original.
* Coalition formation: each task requires its role-typed coalition
  (AGV/Picker counts from the package class); members are chosen to minimise
  the synchronised coalition completion time (latest arrival + handling +
  delivery leg), the analogue of the original's coalition-size/synchronisation
  constraints.
* Battery constraints: a member is only inserted if it can finish the task
  and still reach a charger above a margin; otherwise a recharge directive is
  emitted, mirroring the original's recharge-task insertion.
* Plan repair: delays (mission stuck beyond a factor of its estimated
  completion) and battery aborts vacate coalition slots; repair refills only
  the vacant slot from the free pool instead of re-solving, following the
  original's repair-over-replan principle. Unrepairable coalitions are
  released and their task returns to the allocation pool naturally.
"""

from itertools import combinations
from typing import Dict, List

from tarware.heuristic import AssignmentContext, ChargeDirective, CoalitionCommitment

from .task_model import (
    BasePlanner,
    Task,
    battery_feasible,
    build_tasks,
    cached_travel,
    coalition_completion_steps,
    update_request_ages,
)

# Per-role candidate shortlist size for coalition search; the evaluated fleet
# is 4 AGVs + 2 Pickers, so this is effectively exhaustive.
_TOP_K = 3


class _Record:
    """Planner-side record of a committed coalition (for repair tracking)."""

    __slots__ = ("shelf_id", "agent_ids", "eta", "commit_step", "n_agvs", "n_pickers")

    def __init__(self, shelf_id, agent_ids, eta, commit_step, n_agvs, n_pickers):
        self.shelf_id = shelf_id
        self.agent_ids = set(agent_ids)
        self.eta = eta
        self.commit_step = commit_step
        self.n_agvs = n_agvs
        self.n_pickers = n_pickers


class CalvoPlanner(BasePlanner):
    name = "calvo"

    def __init__(
        self,
        battery_margin: float = 10.0,
        delay_factor: float = 1.5,
        delay_slack: int = 20,
        w_reward: float = 1.0,
        w_urgency: float = 0.002,
    ):
        self.battery_margin = float(battery_margin)
        self.delay_factor = float(delay_factor)
        self.delay_slack = int(delay_slack)
        self.w_reward = float(w_reward)
        self.w_urgency = float(w_urgency)

        self._created_step: Dict[int, int] = {}
        self._records: Dict[int, _Record] = {}   # shelf_id -> record
        self._vacated: List[int] = []            # shelf_ids with aborted members
        self._last_signature = None
        self._stats = {
            "commitments": 0,
            "repairs": 0,
            "repair_failures": 0,
            "releases": 0,
            "recharge_directives": 0,
            "plan_times_s": [],
        }

    # ── event hooks (assignment_listener) ───────────────────────────────
    def on_event(self, event: str, **kw) -> None:
        if event == "abort_charge":
            agent = kw.get("agent")
            if agent is None:
                return
            for rec in self._records.values():
                if int(agent.id) in rec.agent_ids:
                    rec.agent_ids.discard(int(agent.id))
                    if rec.shelf_id not in self._vacated:
                        self._vacated.append(rec.shelf_id)
        elif event == "delivery_event":
            shelf_id = kw.get("shelf_id")
            if shelf_id is not None:
                self._records.pop(int(shelf_id), None)
        elif event == "mission_complete":
            agent = kw.get("agent")
            if agent is None:
                return
            for rec in self._records.values():
                rec.agent_ids.discard(int(agent.id))

    # ── coalition search ─────────────────────────────────────────────────
    def _allocation_order(self, tasks: List[Task], now: int) -> List[Task]:
        def rho(t: Task) -> float:
            return self.w_reward * t.reward_scale + self.w_urgency * (now - t.created_step)

        return sorted(tasks, key=lambda t: (-rho(t), t.coalition_size, t.shelf_id))

    def _best_coalition(self, env, task, free_agvs, free_pickers, cache):
        """Cheapest feasible coalition by synchronised completion time."""

        def shortlist(pool):
            feas = [
                a for a in pool
                if battery_feasible(env, a, task, self.battery_margin, cache)
            ]
            feas.sort(key=lambda a: cached_travel(env, a, task.yx, cache))
            return feas[: max(_TOP_K, task.coalition_size)]

        agv_pool = shortlist(free_agvs) if task.n_agvs else []
        picker_pool = shortlist(free_pickers) if task.n_pickers else []
        if len(agv_pool) < task.n_agvs or len(picker_pool) < task.n_pickers:
            return None

        best = None
        for agvs in combinations(agv_pool, task.n_agvs) if task.n_agvs else [()]:
            for pickers in (
                combinations(picker_pool, task.n_pickers) if task.n_pickers else [()]
            ):
                eta = coalition_completion_steps(env, task, list(agvs), list(pickers), cache)
                if best is None or eta < best[0]:
                    best = (eta, list(agvs), list(pickers))
        return best

    # ── plan repair ──────────────────────────────────────────────────────
    def _repair(self, env, ctx, tasks_by_shelf, free_agvs, free_pickers, cache):
        """Refill vacated slots; release coalitions that cannot be repaired."""
        commitments = []
        still_vacated: List[int] = []
        for shelf_id in self._vacated:
            rec = self._records.get(shelf_id)
            task = tasks_by_shelf.get(shelf_id)
            if rec is None:
                continue
            holders = {
                int(a.id) for a, sid in ctx.assigned_items.items() if int(sid) == shelf_id
            }
            missing = (rec.n_agvs + rec.n_pickers) - len(holders)
            if missing <= 0:
                continue
            if task is None or not holders:
                # Delivered meanwhile, or whole coalition gone: release.
                self._records.pop(shelf_id, None)
                self._stats["releases"] += 1
                continue
            # Determine vacancy by role from current holders, then refill
            # with the nearest feasible agents of the right type.
            holder_agvs = sum(
                1 for a in ctx.agvs if int(a.id) in holders
            )
            holder_pickers = sum(
                1 for p in ctx.pickers if int(p.id) in holders
            )
            need_agvs = rec.n_agvs - holder_agvs
            need_pickers = rec.n_pickers - holder_pickers
            repl_agvs, repl_pickers = [], []
            ok = True
            for need, pool, out in (
                (need_agvs, free_agvs, repl_agvs),
                (need_pickers, free_pickers, repl_pickers),
            ):
                for _ in range(max(0, need)):
                    cands = [
                        a for a in pool
                        if a not in out
                        and battery_feasible(env, a, task, self.battery_margin, cache)
                    ]
                    if not cands:
                        ok = False
                        break
                    out.append(
                        min(cands, key=lambda a: cached_travel(env, a, task.yx, cache))
                    )
                if not ok:
                    break
            if not ok:
                self._stats["repair_failures"] += 1
                still_vacated.append(shelf_id)
                continue
            if repl_agvs or repl_pickers:
                commitments.append(
                    CoalitionCommitment(
                        item=task.shelf,
                        item_loc_id=task.loc_id,
                        req_agvs=task.n_agvs,
                        req_pickers=task.n_pickers,
                        agvs=repl_agvs,
                        pickers=repl_pickers,
                        reason="calvo_plan_repair",
                    )
                )
                rec.agent_ids.update(int(a.id) for a in repl_agvs + repl_pickers)
                for a in repl_agvs:
                    free_agvs.remove(a)
                for p in repl_pickers:
                    free_pickers.remove(p)
                self._stats["repairs"] += 1
        self._vacated = still_vacated
        return commitments

    def _detect_delays(self, ctx) -> None:
        """Mark coalitions exceeding their expected completion as vacated."""
        active_shelves = set(int(s) for s in ctx.assigned_items.values())
        for shelf_id, rec in list(self._records.items()):
            if shelf_id not in active_shelves:
                # Nobody is working on it anymore and no delivery event fired:
                # drop the record; the task re-enters the allocation pool.
                if ctx.timestep > rec.commit_step + rec.eta:
                    self._records.pop(shelf_id, None)
                continue
            deadline = rec.commit_step + self.delay_factor * rec.eta + self.delay_slack
            if ctx.timestep > deadline and shelf_id not in self._vacated:
                self._vacated.append(shelf_id)

    # ── planner interface ────────────────────────────────────────────────
    def replan(self, env, ctx: AssignmentContext) -> list:
        update_request_ages(ctx, self._created_step)
        tasks = build_tasks(env, ctx, self._created_step)
        tasks_by_shelf = {t.shelf_id: t for t in tasks}

        cache: dict = {}
        free_agvs = list(ctx.available_agvs)
        free_pickers = list(ctx.free_pickers)
        commitments: List = []

        # 1. Plan repair for delayed / vacated coalitions.
        self._detect_delays(ctx)
        if self._vacated:
            # Repair targets shelves that are still being worked on, so build
            # their Task records even though they are filtered as "assigned".
            assigned_tasks = {}
            assigned_ids = set(int(s) for s in ctx.assigned_items.values())
            for item in ctx.request_queue:
                if int(item.id) in assigned_ids and int(item.id) in self._vacated:
                    loc_id = ctx.location_id(item.y, item.x)
                    if loc_id is None:
                        continue
                    assigned_tasks[int(item.id)] = Task(
                        shelf=item,
                        shelf_id=int(item.id),
                        package_type=item.package_type,
                        loc_id=int(loc_id),
                        yx=(int(item.y), int(item.x)),
                        created_step=self._created_step.get(int(item.id), ctx.timestep),
                        n_agvs=int(getattr(item, "required_agvs", 1)),
                        n_pickers=int(getattr(item, "required_pickers", 1)),
                        reward_scale=1.0,
                    )
            commitments.extend(
                self._repair(
                    env, ctx, {**tasks_by_shelf, **assigned_tasks},
                    free_agvs, free_pickers, cache,
                )
            )

        # 2. Greedy insertion in reward-coefficient order. Skipped when the
        #    free-agent set and the unassigned task set are unchanged since the
        #    last insertion pass (event-triggered receding horizon; repair
        #    above still runs every step).
        signature = (
            frozenset(int(a.id) for a in free_agvs),
            frozenset(int(p.id) for p in free_pickers),
            frozenset(t.shelf_id for t in tasks),
        )
        if signature == self._last_signature and not commitments:
            return []
        self._last_signature = signature

        for task in self._allocation_order(tasks, ctx.timestep):
            if not free_agvs and not free_pickers:
                break
            best = self._best_coalition(env, task, free_agvs, free_pickers, cache)
            if best is None:
                continue
            eta, agvs, pickers = best
            commitments.append(
                CoalitionCommitment(
                    item=task.shelf,
                    item_loc_id=task.loc_id,
                    req_agvs=task.n_agvs,
                    req_pickers=task.n_pickers,
                    agvs=agvs,
                    pickers=pickers,
                    reason="calvo_greedy_insertion",
                )
            )
            self._records[task.shelf_id] = _Record(
                task.shelf_id,
                [int(a.id) for a in agvs + pickers],
                int(eta),
                ctx.timestep,
                task.n_agvs,
                task.n_pickers,
            )
            for a in agvs:
                free_agvs.remove(a)
            for p in pickers:
                free_pickers.remove(p)

        # 3. Proactive recharge insertion for agents that cannot serve any
        #    pending task within battery margin.
        if tasks:
            for agent in free_agvs + free_pickers:
                if not any(
                    battery_feasible(env, agent, t, self.battery_margin, cache)
                    for t in tasks
                ):
                    commitments.append(
                        ChargeDirective(agent, reason="calvo_recharge_insertion")
                    )
                    self._stats["recharge_directives"] += 1

        self._stats["commitments"] += sum(
            1 for c in commitments if isinstance(c, CoalitionCommitment)
        )
        return commitments

    def stats(self) -> dict:
        s = dict(self._stats)
        s.pop("plan_times_s", None)
        return s
