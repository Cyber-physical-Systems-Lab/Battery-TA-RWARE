"""scipy/HiGHS MILP encoding for per-batch coalition assignment (BMRTA).

Solves one batch of coalition tasks: every task j exposes role-typed slots
(j, r); binary x[i, (j, r)] assigns agent i to a slot, binary y_j marks task j
as fully covered, and continuous M_j is the coalition makespan (latest member
arrival). Slot coverage is tied to y_j so a task is either fully crewed or
left for a later replan, and the makespan link encodes the Picker-before-AGV
load synchronisation (the AGV cannot load before the last member arrives).

    min  sum_j  w_makespan * M_j + w_sumcost * sum_i t_ij x_ijr
         - sum_j reward_j * y_j
    s.t. sum_i x_ijr = y_j                (every slot of j covered iff y_j)
         sum_jr x_ijr <= 1               (agent serves at most one slot)
         M_j >= sum_i t_ij x_ijr         (per slot: makespan >= arrival)

Falls back to a greedy nearest-feasible completion when HiGHS fails or the
time limit is hit, so the planner never stalls the episode.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def solve_assignment_milp(
    slot_candidates: List[Dict[int, float]],
    slot_task: List[int],
    n_tasks: int,
    task_reward: List[float],
    n_agents: int,
    w_makespan: float = 1.0,
    w_sumcost: float = 0.1,
    time_limit_s: float = 2.0,
) -> Optional[List[Tuple[int, int]]]:
    """Solve one batch; returns [(slot_idx, agent_idx)] or None on failure.

    Parameters
    ----------
    slot_candidates : per-slot {agent_idx: travel_steps} for feasible agents.
    slot_task : task index of each slot.
    task_reward : per-task objective bonus for full coverage; must dominate
        the cost terms so that covering a task always beats skipping it.
    """
    nx = sum(len(c) for c in slot_candidates)
    ns = len(slot_candidates)
    if nx == 0 or ns == 0:
        return []

    # Variable layout: [x_0..x_{nx-1} | y_0..y_{nt-1} | M_0..M_{nt-1}]
    x_index: List[Tuple[int, int]] = []   # var -> (slot_idx, agent_idx)
    x_cost: List[float] = []
    for s, cands in enumerate(slot_candidates):
        for agent_idx, t in sorted(cands.items()):
            x_index.append((s, agent_idx))
            x_cost.append(float(t))
    nt = n_tasks
    nvar = nx + 2 * nt
    y_off, m_off = nx, nx + nt

    t_max = max(x_cost) if x_cost else 1.0

    c = np.zeros(nvar)
    c[:nx] = w_sumcost * np.asarray(x_cost)
    for j in range(nt):
        c[y_off + j] = -float(task_reward[j])
        c[m_off + j] = w_makespan

    integrality = np.zeros(nvar)
    integrality[: nx + nt] = 1

    lb = np.zeros(nvar)
    ub = np.ones(nvar)
    ub[m_off:] = t_max + 1.0

    constraints = []

    # Slot coverage: sum_i x_ijr - y_j = 0
    a_cov = np.zeros((ns, nvar))
    for v, (s, _i) in enumerate(x_index):
        a_cov[s, v] = 1.0
    for s in range(ns):
        a_cov[s, y_off + slot_task[s]] = -1.0
    constraints.append(LinearConstraint(a_cov, 0.0, 0.0))

    # Agent capacity: sum_jr x_ijr <= 1
    a_cap = np.zeros((n_agents, nvar))
    for v, (_s, i) in enumerate(x_index):
        a_cap[i, v] = 1.0
    constraints.append(LinearConstraint(a_cap, -np.inf, 1.0))

    # Makespan link per slot: M_j - sum_i t_ij x_ijr >= 0
    a_mk = np.zeros((ns, nvar))
    for v, (s, _i) in enumerate(x_index):
        a_mk[s, v] = -x_cost[v]
    for s in range(ns):
        a_mk[s, m_off + slot_task[s]] = 1.0
    constraints.append(LinearConstraint(a_mk, 0.0, np.inf))

    res = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        options={"time_limit": float(time_limit_s), "presolve": True},
    )
    if res is None or res.status != 0 or res.x is None:
        return None

    chosen: List[Tuple[int, int]] = []
    covered = {j for j in range(nt) if res.x[y_off + j] > 0.5}
    for v, (s, i) in enumerate(x_index):
        if res.x[v] > 0.5 and slot_task[s] in covered:
            chosen.append((s, i))
    return chosen


def greedy_fallback(
    slot_candidates: List[Dict[int, float]],
    slot_task: List[int],
    n_tasks: int,
    task_priority: List[float],
) -> List[Tuple[int, int]]:
    """Nearest-feasible completion: crew tasks in priority order, all-or-nothing."""
    chosen: List[Tuple[int, int]] = []
    used: set = set()
    order = sorted(range(n_tasks), key=lambda j: -task_priority[j])
    for j in order:
        slots = [s for s in range(len(slot_task)) if slot_task[s] == j]
        picks: List[Tuple[int, int]] = []
        local_used: set = set()
        ok = True
        for s in slots:
            cands = [
                (t, i) for i, t in slot_candidates[s].items()
                if i not in used and i not in local_used
            ]
            if not cands:
                ok = False
                break
            t, i = min(cands)
            picks.append((s, i))
            local_used.add(i)
        if ok and picks:
            chosen.extend(picks)
            used.update(local_used)
    return chosen


def timed_solve(
    *args,
    **kwargs,
) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    """`solve_assignment_milp` wrapped with wall-clock timing."""
    t0 = time.perf_counter()
    result = solve_assignment_milp(*args, **kwargs)
    return result, time.perf_counter() - t0
