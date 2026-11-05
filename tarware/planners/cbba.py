r"""Consensus-Based decentralized auction for cross-role coalition allocation.

Reimplementation of the Consensus-Based Bundle/Auction Algorithm
(CBBA/CBAA, Choi et al., "Consensus-Based Decentralized Auctions for Robust
Task Allocation," IEEE T-RO 2009), adapted to the heterogeneous warehouse.
Unlike the centralized BMRTA / Calvo planners, CBBA is genuinely
\emph{decentralized}: each agent scores tasks from its \emph{own} local state
(its position and battery only), and conflicts are resolved by max-consensus on
per-slot winning bids rather than by a central optimizer. Because the executor
commits one mission per agent at a time, the bundle reduces to a single
role-slot, i.e. the consensus-based auction (CBAA) variant.

Coalition tasks (STANDARD = 1 AGV + 1 Picker, HEAVY = 1 AGV + 2 Pickers) are
decomposed into role slots; agents bid only on type-compatible slots, and a
request is committed only when every one of its slots has a free winning agent
(coalition synchronization). Agents that cannot battery-feasibly bid on any
pending task are sent to charge.
"""

from typing import Dict

from tarware.heuristic import AssignmentContext, ChargeDirective, CoalitionCommitment

from .task_model import (
    BasePlanner,
    battery_feasible,
    build_tasks,
    cached_travel,
    update_request_ages,
)


class CBBAPlanner(BasePlanner):
    name = "cbba"

    def __init__(self, battery_margin: float = 10.0, max_rounds: int = 20,
                 w_urgency: float = 0.002):
        self.battery_margin = float(battery_margin)
        self.max_rounds = int(max_rounds)
        self.w_urgency = float(w_urgency)
        self._created_step: Dict[int, int] = {}
        self._last_signature = None
        self._stats = {"commitments": 0, "auction_rounds": 0,
                       "recharge_directives": 0, "replans": 0}

    @staticmethod
    def _bid(env, agent, task, cache) -> float:
        """Local bid: higher for higher-value, nearer tasks. Uses only the
        agent's own position (decentralized score, no global optimization)."""
        d = cached_travel(env, agent, task.yx, cache)
        return float(task.reward_scale) / (1.0 + float(d))

    def replan(self, env, ctx: AssignmentContext) -> list:
        update_request_ages(ctx, self._created_step)
        tasks = build_tasks(env, ctx, self._created_step)
        free_agvs = list(ctx.available_agvs)
        free_pickers = list(ctx.free_pickers)
        if not tasks or (not free_agvs and not free_pickers):
            return []

        # Re-solve only when the free set or pending tasks changed.
        sig = (frozenset(int(a.id) for a in free_agvs),
               frozenset(int(p.id) for p in free_pickers),
               frozenset(t.shelf_id for t in tasks))
        if sig == self._last_signature:
            return []
        self._last_signature = sig
        self._stats["replans"] += 1

        cache: dict = {}
        # Each agent scores every feasible, type-compatible task from its OWN
        # local state (position+battery) -> bids[aid][task_idx]. A task needs a
        # full role-typed coalition, so winner determination is coalition-aware:
        # the consensus iteratively commits the fully-crewable task with the
        # highest aggregate bid, crewed by its top role-bidders, until no task
        # can be crewed. This keeps CBBA's decentralized local bidding while
        # avoiding coalition fragmentation (independent slot bids would scatter
        # agents across tasks and complete none).
        free_agv = {int(a.id): a for a in free_agvs}
        free_pick = {int(p.id): p for p in free_pickers}
        bids: Dict[int, Dict[int, float]] = {}
        for pool, is_agv in ((free_agv, True), (free_pick, False)):
            for aid, agent in pool.items():
                d = {}
                for j, t in enumerate(tasks):
                    need = t.n_agvs if is_agv else t.n_pickers
                    if need <= 0:
                        continue
                    if not battery_feasible(env, agent, t, self.battery_margin, cache):
                        continue
                    d[j] = self._bid(env, agent, t, cache)
                bids[aid] = d

        commitments = []
        committed_agents = set()
        remaining = set(range(len(tasks)))
        while remaining:
            self._stats["auction_rounds"] += 1
            best_j, best_crew, best_val = None, None, -1.0
            for j in remaining:
                t = tasks[j]
                agv_bidders = sorted(
                    ((bids[a][j], a) for a in free_agv
                     if a not in committed_agents and j in bids.get(a, {})),
                    reverse=True)
                pick_bidders = sorted(
                    ((bids[p][j], p) for p in free_pick
                     if p not in committed_agents and j in bids.get(p, {})),
                    reverse=True)
                if len(agv_bidders) < t.n_agvs or len(pick_bidders) < t.n_pickers:
                    continue
                crew_a = agv_bidders[:t.n_agvs]
                crew_p = pick_bidders[:t.n_pickers]
                val = sum(b for b, _ in crew_a) + sum(b for b, _ in crew_p)
                if val > best_val:
                    best_val = val
                    best_j = j
                    best_crew = ([free_agv[a] for _, a in crew_a],
                                 [free_pick[p] for _, p in crew_p])
            if best_j is None:
                break
            t = tasks[best_j]
            agvs, pickers = best_crew
            commitments.append(CoalitionCommitment(
                item=t.shelf, item_loc_id=t.loc_id,
                req_agvs=t.n_agvs, req_pickers=t.n_pickers,
                agvs=agvs, pickers=pickers,
                reason="cbba_consensus_auction"))
            committed_agents.update(int(a.id) for a in agvs + pickers)
            remaining.discard(best_j)
        self._stats["commitments"] += len(commitments)

        # Agents with no battery-feasible bid on any pending task -> charge.
        for pool in (free_agv, free_pick):
            for aid, agent in pool.items():
                if aid in committed_agents:
                    continue
                if not any(battery_feasible(env, agent, t, self.battery_margin, cache)
                           for t in tasks):
                    commitments.append(ChargeDirective(agent, reason="cbba_recharge"))
                    self._stats["recharge_directives"] += 1
        return commitments

    def stats(self) -> dict:
        return dict(self._stats)
