"""BMRTA-style batch task allocation adapted to the online warehouse.

Reimplementation of the batching strategy of Zhang et al., "Energy Efficient
Multi-Robot Task Allocation Constrained by Time Window and Precedence"
(IEEE T-ASE 2023; repo TWPC-MRTA), adapted as a receding-horizon planner:

* The original solves a static instance (tasks with earliest start times,
  time-window lengths, and cross-task precedence) offline with a per-batch
  Gurobi MIP. Here the planner is re-invoked at event-triggered replan
  boundaries on the current request queue.
* All queued requests are immediately startable (EST = now) and carry no
  cross-task precedence, so the time-window component of the original measure
  graph degenerates to request age and precedence reduces to the *within-task*
  Picker-before-AGV-load synchronisation, encoded by the coalition-makespan
  variable of the batch MILP (see ``milp_utils``).
* The measure graph reflects task spatial sparsity (Manhattan distance between
  shelf cells); sparsification + connected components yield batches, which are
  solved oldest-first with scipy/HiGHS instead of Gurobi.
* Heterogeneous coalition slots (AGV/Picker per package type) extend the
  original single-robot tasks; this is required by the AGV-Picker domain.
"""

from typing import Dict, List

import numpy as np

from tarware.definitions import AgentType
from tarware.heuristic import AssignmentContext, ChargeDirective, CoalitionCommitment

from .milp_utils import greedy_fallback, timed_solve
from .task_model import (
    BasePlanner,
    Task,
    battery_feasible,
    build_tasks,
    cached_travel,
    update_request_ages,
)


class BMRTAPlanner(BasePlanner):
    name = "bmrta"

    def __init__(
        self,
        batch_size_max: int = 8,
        milp_time_limit_s: float = 2.0,
        w_makespan: float = 1.0,
        w_sumcost: float = 0.1,
        sparsity_quantile: float = 0.5,
        battery_margin: float = 10.0,
    ):
        self.batch_size_max = int(batch_size_max)
        self.milp_time_limit_s = float(milp_time_limit_s)
        self.w_makespan = float(w_makespan)
        self.w_sumcost = float(w_sumcost)
        self.sparsity_quantile = float(sparsity_quantile)
        self.battery_margin = float(battery_margin)

        self._created_step: Dict[int, int] = {}
        self._last_signature = None
        self._stats = {
            "milp_calls": 0,
            "milp_failures": 0,
            "fallback_calls": 0,
            "solve_times_s": [],
            "commitments": 0,
            "batches": 0,
        }

    # ── batching ─────────────────────────────────────────────────────────
    @staticmethod
    def _manhattan(a: Task, b: Task) -> int:
        return abs(a.yx[0] - b.yx[0]) + abs(a.yx[1] - b.yx[1])

    def _batch(self, tasks: List[Task]) -> List[List[Task]]:
        """Measure-graph batching: sparsify pairwise distances, split components."""
        if len(tasks) <= 1:
            return [tasks] if tasks else []
        weights = [
            self._manhattan(tasks[i], tasks[j])
            for i in range(len(tasks))
            for j in range(i + 1, len(tasks))
        ]
        threshold = float(np.quantile(weights, self.sparsity_quantile))

        # Union-find over tasks connected by below-threshold edges.
        parent = list(range(len(tasks)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                if self._manhattan(tasks[i], tasks[j]) <= threshold:
                    parent[find(i)] = find(j)

        comps: Dict[int, List[Task]] = {}
        for i, t in enumerate(tasks):
            comps.setdefault(find(i), []).append(t)

        batches: List[List[Task]] = []
        for comp in comps.values():
            comp = sorted(comp, key=lambda t: t.created_step)
            for k in range(0, len(comp), self.batch_size_max):
                batches.append(comp[k : k + self.batch_size_max])
        batches.sort(key=lambda b: min(t.created_step for t in b))
        return batches

    # ── per-batch solving ────────────────────────────────────────────────
    def _solve_batch(self, env, batch, free_agvs, free_pickers, cache):
        """Returns [(task, agvs, pickers)] for fully-crewed coalitions."""
        agents = list(free_agvs) + list(free_pickers)
        if not agents or not batch:
            return []
        agent_idx = {int(a.id): k for k, a in enumerate(agents)}

        slot_candidates: List[Dict[int, float]] = []
        slot_task: List[int] = []
        slot_role: List[AgentType] = []
        for j, task in enumerate(batch):
            for role, count in (
                (AgentType.AGV, task.n_agvs),
                (AgentType.PICKER, task.n_pickers),
            ):
                pool = free_agvs if role == AgentType.AGV else free_pickers
                for _ in range(count):
                    cands = {
                        agent_idx[int(a.id)]: float(cached_travel(env, a, task.yx, cache))
                        for a in pool
                        if battery_feasible(env, a, task, self.battery_margin, cache)
                    }
                    slot_candidates.append(cands)
                    slot_task.append(j)
                    slot_role.append(role)

        if not slot_candidates:
            return []

        t_max = max((max(c.values()) for c in slot_candidates if c), default=1.0)
        rewards = [10.0 * t_max * task.reward_scale for task in batch]

        self._stats["milp_calls"] += 1
        chosen, dt = timed_solve(
            slot_candidates,
            slot_task,
            len(batch),
            rewards,
            len(agents),
            w_makespan=self.w_makespan,
            w_sumcost=self.w_sumcost,
            time_limit_s=self.milp_time_limit_s,
        )
        self._stats["solve_times_s"].append(round(dt, 6))
        if chosen is None:
            self._stats["milp_failures"] += 1
            self._stats["fallback_calls"] += 1
            priority = [task.reward_scale for task in batch]
            chosen = greedy_fallback(slot_candidates, slot_task, len(batch), priority)

        by_task: Dict[int, Dict[str, list]] = {}
        for s, i in chosen:
            j = slot_task[s]
            rec = by_task.setdefault(j, {"agvs": [], "pickers": []})
            agent = agents[i]
            if slot_role[s] == AgentType.AGV:
                rec["agvs"].append(agent)
            else:
                rec["pickers"].append(agent)

        out = []
        for j, rec in by_task.items():
            task = batch[j]
            if len(rec["agvs"]) == task.n_agvs and len(rec["pickers"]) == task.n_pickers:
                out.append((task, rec["agvs"], rec["pickers"]))
        return out

    # ── planner interface ────────────────────────────────────────────────
    def replan(self, env, ctx: AssignmentContext) -> list:
        update_request_ages(ctx, self._created_step)
        tasks = build_tasks(env, ctx, self._created_step)
        if not tasks or (not ctx.available_agvs and not ctx.free_pickers):
            return []

        signature = (
            frozenset(int(a.id) for a in ctx.available_agvs),
            frozenset(int(p.id) for p in ctx.free_pickers),
            frozenset(t.shelf_id for t in tasks),
        )
        if signature == self._last_signature:
            return []
        self._last_signature = signature

        cache: dict = {}
        free_agvs = list(ctx.available_agvs)
        free_pickers = list(ctx.free_pickers)
        commitments = []
        for batch in self._batch(tasks):
            self._stats["batches"] += 1
            if not free_agvs and not free_pickers:
                break
            for task, agvs, pickers in self._solve_batch(
                env, batch, free_agvs, free_pickers, cache
            ):
                commitments.append(
                    CoalitionCommitment(
                        item=task.shelf,
                        item_loc_id=task.loc_id,
                        req_agvs=task.n_agvs,
                        req_pickers=task.n_pickers,
                        agvs=agvs,
                        pickers=pickers,
                        reason="bmrta_batch_milp",
                    )
                )
                for a in agvs:
                    free_agvs.remove(a)
                for p in pickers:
                    free_pickers.remove(p)
        # Agents that cannot battery-feasibly serve any pending task would idle
        # on standby drain until the low-battery override; send them to charge.
        for agent in free_agvs + free_pickers:
            if not any(
                battery_feasible(env, agent, t, self.battery_margin, cache)
                for t in tasks
            ):
                commitments.append(
                    ChargeDirective(agent, reason="bmrta_recharge_infeasible")
                )

        self._stats["commitments"] += len(commitments)
        return commitments

    def stats(self) -> dict:
        s = dict(self._stats)
        times = s.pop("solve_times_s")
        s["solve_time_median_ms"] = round(1e3 * float(np.median(times)), 3) if times else 0.0
        s["solve_time_max_ms"] = round(1e3 * float(np.max(times)), 3) if times else 0.0
        return s
