"""
Heuristic episode runner for the current Tarware environment.

The controller converts high-level warehouse missions into macro actions for
heterogeneous AGV and picker teams. It supports SOLO, STANDARD, and LARGE
package requirements, outbound delivery and inbound shelf return cycles, and
explicit charging-station missions.

The runner also wires in the research instrumentation used by experiments:
independent or synchronized picker-AGV charging, fixed-interval or
energy-deviation replanning, Stage-2 weight-revelation checks, coupling
metrics, optional GIF capture, and role-emergence summaries for AGV mission
specialization.
"""

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import imageio

from tarware.utils.utils import flatten_list, split_list
from tarware.definitions import PackageType, PackageDirection, CollisionLayers
from tarware.warehouse import Agent, AgentType, _BATTERY_FULL
from tarware.energy_coupling import (
    ChargingStrategy,
    CouplingMetrics,
    EnergyModel,
    JointChargingCoordinator,
    PickerAGVPair,
)
from tarware.replanning import (
    ReplanningPolicy,
    ReplanningController,
    ReplanningMetrics,
)
from tarware.role_assignment import RoleEmergenceTracker, RoleEmergenceMetrics


class MissionType(Enum):
    PICKING = 1
    RETURNING = 2
    DELIVERING = 3
    CHARGING = 4


@dataclass
class Mission:
    mission_type: MissionType
    location_id: int
    location_x: int
    location_y: int
    assigned_time: int
    at_location: bool = False


@dataclass
class AssignmentContext:
    """Read-only snapshot handed to an assignment function each step.

    ``available_agvs``/``free_pickers`` carry the executor's single source of
    truth for availability (not busy, not carrying, not assigned, not failed,
    battery above threshold) so every planner sees identical semantics.
    """

    request_queue: list
    agvs: list
    pickers: list
    available_agvs: list
    free_pickers: list
    assigned_agvs: dict
    assigned_pickers: dict
    assigned_items: dict
    carried_shelf_ids: set
    failed_active: set
    low_battery_threshold: float
    timestep: int
    location_id: callable
    log_behavior: callable


@dataclass
class CoalitionCommitment:
    """A planner's decision to start one request with a full coalition now."""

    item: object            # shelf from the request queue
    item_loc_id: int        # macro-action id of the shelf cell
    req_agvs: int
    req_pickers: int
    agvs: list
    pickers: list
    reason: str = "best_feasible_request"


@dataclass
class ChargeDirective:
    """A planner's decision to proactively send an idle agent to charge."""

    agent: object
    reason: str = "planner_recharge"


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _path_len(env, start_agent: Agent, target_yx) -> int:
    """Compute A* path length from agent position to target (y, x)."""
    path = env.find_path(
        (start_agent.y, start_agent.x), target_yx, start_agent, care_for_agents=False
    )
    return len(path) if path else int(1e6)


def _nearest_charger_path_len(env, agent: Agent) -> int:
    """Minimum path length from agent to any charging station."""
    return min(
        _path_len(env, agent, (s.y, s.x)) for s in env.charging_stations
    )


def _choose_goal(env, agent: Agent, goal_locations, claimed: set):
    """Nearest goal not already claimed by another DELIVERING mission.

    Sending every carrier to its globally nearest goal funnels the whole team
    into one dock and produces unrecoverable multi-agent standoffs at the dock
    corner (the env clash logic only resolves pairwise conflicts). Preferring
    an unclaimed dock disperses traffic; if every dock is claimed, fall back
    to the globally nearest one.
    """
    ranked = sorted(goal_locations, key=lambda g: _path_len(env, agent, (g[1], g[0])))
    for goal in ranked:
        if (goal[0], goal[1]) not in claimed:
            return goal
    return ranked[0]


def _claimed_goal_cells(assigned_agvs: dict, assigned_pickers: dict) -> set:
    return {
        (m.location_x, m.location_y)
        for m in list(assigned_agvs.values()) + list(assigned_pickers.values())
        if m.mission_type == MissionType.DELIVERING
    }


def _nearest_charger(env, agent: Agent, occupied: set, assigned: set):
    """Find nearest available charging station."""
    coords_to_id = {v: k for k, v in env.action_id_to_coords_map.items()}
    best, best_len = None, float("inf")
    for s in env.charging_stations:
        if (s.x, s.y) in occupied or (s.x, s.y) in assigned:
            continue
        d = _path_len(env, agent, (s.y, s.x))
        if d < best_len:
            best_len, best = d, s
    if best is None:
        return None, None
    return best, coords_to_id.get((best.y, best.x))


# ────────────────────────────────────────────────────────────────────────────
# Default assignment function (nearest-feasible rule baseline)
# ────────────────────────────────────────────────────────────────────────────

def greedy_assignment(env, ctx: AssignmentContext) -> list:
    """Nearest-feasible request selection (the original rule baseline).

    Commits at most one coalition per call: for each unassigned request with
    enough free agents, picks the closest agents per role and scores the plan
    by ``100 * coalition_size + total_travel``; the cheapest plan wins.
    """
    available_agvs = ctx.available_agvs
    free_pickers = ctx.free_pickers
    assigned_shelf_ids = set(ctx.assigned_items.values())

    candidate_plans = []
    for item in ctx.request_queue:
        req_agvs = int(getattr(item, "required_agvs", 1))
        req_pickers = int(getattr(item, "required_pickers", 1))
        item_loc_id = ctx.location_id(item.y, item.x)

        if item_loc_id is None:
            continue
        if item.id in assigned_shelf_ids or item.id in ctx.carried_shelf_ids:
            continue
        if len(available_agvs) < req_agvs or len(free_pickers) < req_pickers:
            continue

        agv_order = np.argsort([
            _path_len(env, a, (item.y, item.x)) for a in available_agvs
        ]) if available_agvs else []
        picker_order = np.argsort([
            _path_len(env, p, (item.y, item.x)) for p in free_pickers
        ]) if free_pickers else []
        chosen_agvs = [available_agvs[i] for i in list(agv_order)[:req_agvs]]
        chosen_pickers = [free_pickers[i] for i in list(picker_order)[:req_pickers]]
        travel_cost = sum(_path_len(env, a, (item.y, item.x)) for a in chosen_agvs)
        travel_cost += sum(_path_len(env, p, (item.y, item.x)) for p in chosen_pickers)
        complexity_cost = 100.0 * (req_agvs + req_pickers)
        candidate_plans.append(
            (
                complexity_cost + float(travel_cost),
                item,
                item_loc_id,
                req_agvs,
                req_pickers,
                chosen_agvs,
                chosen_pickers,
            )
        )

    if not candidate_plans:
        first_item = ctx.request_queue[0]
        ctx.log_behavior(
            "planner_wait_resources",
            shelf_id=int(first_item.id),
            package_type=first_item.package_type.name,
            reason="no_feasible_request_in_queue",
        )
        return []

    _, item, item_loc_id, req_agvs, req_pickers, chosen_agvs, chosen_pickers = min(
        candidate_plans, key=lambda x: x[0]
    )
    return [
        CoalitionCommitment(
            item=item,
            item_loc_id=item_loc_id,
            req_agvs=req_agvs,
            req_pickers=req_pickers,
            agvs=chosen_agvs,
            pickers=chosen_pickers,
        )
    ]


# ────────────────────────────────────────────────────────────────────────────
# Core heuristic episode runner
# ────────────────────────────────────────────────────────────────────────────

def heuristic_episode(
    env,
    render: bool = False,
    seed: Optional[int] = None,
    save_gif: bool = False,
    gif_path: str = "episode.gif",
    charging_strategy: ChargingStrategy = ChargingStrategy.INDEPENDENT,
    low_battery_threshold: float = 10.0,
    alpha: float = 0.9,
    # RQ3 replanning parameters
    replanning_policy: ReplanningPolicy = ReplanningPolicy.NO_REPLANNING,
    replan_interval: int = 5,
    deviation_threshold: float = 8.0,
    use_stage2_recheck: bool = True,
    # RQ4 role tracking
    track_roles: bool = True,
    # When False, AGVs/pickers do NOT load the inbound shelf and navigate it
    # back to its home slot after a delivery; they instead end the mission at
    # the goal so they are reassigned to the next outbound shelf. This matches
    # the MARL-trained policy behaviour, which never enters a RETURNING phase.
    return_shelves: bool = True,
    # List of (agent_index, start_step) tuples. Each entry permanently locks the
    # named agent (index into base_env.agents) to action 0 (NOOP) once
    # timestep >= start_step. The agent's current mission is dropped on
    # activation so the heuristic stops trying to bid for it, mirroring the
    # action-channel failure injection used by the learned-policy evaluator.
    failures: Optional[List[Tuple[int, int]]] = None,
    # Pluggable task assignment: callable (env, AssignmentContext) -> list of
    # CoalitionCommitment | ChargeDirective. None -> greedy_assignment (the
    # original nearest-feasible rule baseline, behavior-identical).
    assignment_fn=None,
    # Optional callable (event: str, **kw) notified at mission boundaries
    # (mission_complete, abort_charge, charge_complete, delivery_event) so
    # stateful planners can run plan repair.
    assignment_listener=None,
) -> tuple:
    """
    Run one heuristic episode.

    Parameters
    ----------
    charging_strategy : ChargingStrategy
        INDEPENDENT — each agent charges independently.
        SYNCHRONIZED — paired picker-AGV use joint depletion prediction.
    low_battery_threshold : float
        Battery level below which independent charging is triggered.
    alpha : float
        Safety confidence level for synchronized charging (Stage 1 threshold).
    replanning_policy : ReplanningPolicy
        NO_REPLANNING   — never replan (baseline).
        FIXED_INTERVAL  — replan every `replan_interval` steps (REACH-MP analog).
        EVENT_TRIGGERED — replan when energy deviation exceeds `deviation_threshold`.
    replan_interval : int
        Steps between forced replans (FIXED_INTERVAL only).
    deviation_threshold : float
        Battery units of negative deviation that trigger a replan (EVENT_TRIGGERED).
        Smaller values → more sensitive (risk of oscillation).
        Larger values  → less sensitive (risk of missing genuine depletions).
    use_stage2_recheck : bool
        If True, also run Stage 2 feasibility check when shelf weight is
        revealed (AGV picks up shelf). Used for RQ3b interaction study.
    track_roles : bool
        If True (default), track per-agent mission-type distribution and
        section specialization for RQ4 role emergence analysis.

    Returns
    -------
    (all_infos, global_episode_return, episode_returns, coupling_metrics,
     replanning_metrics, role_metrics)
    coupling_metrics   : dict {agv_id: dict} — empty for NO-REPLANNING baseline.
    replanning_metrics : ReplanningMetrics   — None if NO_REPLANNING.
    role_metrics       : RoleEmergenceMetrics — None if track_roles=False.
    """
    # `gym.make()` returns Gymnasium wrappers (e.g. OrderEnforcing), while the
    # heuristic needs warehouse-specific attributes/methods defined on the base env.
    runtime_env = env
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    env = base_env

    non_goal_location_ids = np.array(base_env.shelf_action_ids)
    location_map = base_env.action_id_to_coords_map
    coords_to_id = {v: k for k, v in location_map.items()}

    def location_id(y, x):
        yx = (int(y), int(x))
        xy = (int(x), int(y))
        return coords_to_id.get(yx, coords_to_id.get(xy))

    _ = runtime_env.reset(seed=seed)
    done = False
    all_infos = []
    timestep = 0

    agents = base_env.agents
    agvs = [a for a in agents if a.type == AgentType.AGV]
    pickers = [a for a in agents if a.type == AgentType.PICKER]

    # ── Failure injection (resilience experiments) ───────────────────────
    failure_specs = list(failures or [])
    failed_starts: Dict[Agent, int] = {}
    for idx, start_step in failure_specs:
        if idx < 0 or idx >= len(agents):
            raise ValueError(
                f"failure agent_index={idx} outside range 0..{len(agents)-1}"
            )
        if int(start_step) < 0:
            raise ValueError(f"failure start_step={start_step} must be non-negative")
        failed_starts[agents[idx]] = int(start_step)
    failed_active: set = set()
    failure_audit: Dict[int, Dict[str, object]] = {
        idx: {"start_step": int(start_step), "override_count": 0, "all_zero": True}
        for idx, start_step in failure_specs
    }
    agent_to_failure_idx: Dict[Agent, int] = {
        agents[idx]: idx for idx, _ in failure_specs
    }

    # Assign pickers to warehouse sections
    sections = base_env.rack_groups
    picker_sections = split_list(sections, len(pickers)) if pickers else []
    picker_sections = [flatten_list(l) for l in picker_sections]

    # ── Picker-AGV pairing for coupling study ────────────────────────────
    # Each section is served by one picker; AGVs picking from that section
    # are "paired" with that picker for energy coupling analysis.
    # We build a static pairing: picker i → section i.
    # Dynamically, we track which picker is currently serving each AGV.
    energy_model = EnergyModel()
    pairs: List[PickerAGVPair] = []
    agv_to_picker: Dict[Agent, Agent] = {}   # dynamic: updated per task assignment

    if pickers and agvs:
        # Round-robin initial pairing (1 picker per section)
        for i, agv in enumerate(agvs):
            paired_picker = pickers[i % len(pickers)]
            pair = PickerAGVPair(
                agv_id=agv.id,
                picker_id=paired_picker.id,
                energy_model=energy_model,
                alpha=alpha,
            )
            pairs.append(pair)
            agv_to_picker[agv] = paired_picker

    coordinator = JointChargingCoordinator(
        pairs=pairs,
        strategy=charging_strategy,
        low_battery_threshold=low_battery_threshold,
    )

    # ── RQ3: Replanning controller ───────────────────────────────────────
    replan_ctrl = ReplanningController(
        policy=replanning_policy,
        energy_model=energy_model,
        replan_interval=replan_interval,
        deviation_threshold=deviation_threshold,
        use_stage2_recheck=use_stage2_recheck,
    )

    # ── RQ4: Role emergence tracker ──────────────────────────────────────
    role_tracker = (
        RoleEmergenceTracker(
            agv_ids=[a.id for a in agvs],
            num_pickers=len(pickers),
        )
        if track_roles
        else None
    )

    # ── Mission tracking ─────────────────────────────────────────────────
    assigned_agvs: Dict[Agent, Mission] = OrderedDict()
    assigned_pickers: Dict[Agent, Mission] = OrderedDict()
    assigned_items: Dict[Agent, int] = OrderedDict()
    # Shelf IDs where required_pickers==0 (SOLO); populated once per episode.
    solo_shelf_ids: set = {s.id for s in base_env.shelfs if getattr(s, "required_pickers", 1) == 0}
    # LARGE tasks: shelf_id → helper AGV (freed once lead lifts shelf).
    large_group_helpers: Dict[int, Agent] = {}

    global_episode_return = 0.0
    episode_returns = np.zeros(base_env.num_agents)
    frames = []
    total_deliveries = 0
    behavior_events = []

    # Liveness circuit breaker: en-route agents that have not moved for this
    # many steps are in an unresolvable multi-agent standoff (the env clash
    # logic skips 2-cycle swaps, and turn-oscillating agents evade its stuck
    # counter). They get re-targeted or force re-pathed below.
    STUCK_RELEASE_STEPS = 40
    stagnation: Dict[Agent, Tuple[tuple, int]] = {}

    def _agent_ids(items: List[Agent]) -> List[int]:
        return [int(a.id - 1) for a in items]

    def log_behavior(event_type: str, **payload) -> None:
        behavior_events.append(
            {
                "episode_step": int(timestep),
                "event_type": event_type,
                **payload,
            }
        )

    def _notify(event: str, **kw) -> None:
        if assignment_listener is not None:
            assignment_listener(event, timestep=timestep, **kw)

    def _assigned_charging_stations() -> set:
        coords = set()
        for m in list(assigned_agvs.values()) + list(assigned_pickers.values()):
            if m.mission_type == MissionType.CHARGING:
                coords.add((m.location_x, m.location_y))
        return coords

    # ── Helper: assign charging mission to an agent ──────────────────────
    def assign_charge(agent: Agent, assigned_dict: dict) -> bool:
        occupied = {(a.x, a.y) for a in agents}
        station, sid = _nearest_charger(env, agent, occupied, _assigned_charging_stations())
        if station is None or sid is None:
            log_behavior(
                "charge_wait_no_station",
                agent_id=int(agent.id - 1),
                agent_type=agent.type.name,
                battery=float(agent.battery),
                reason="no_available_charger",
            )
            return False
        assigned_dict[agent] = Mission(
            MissionType.CHARGING, sid, station.x, station.y, timestep
        )
        log_behavior(
            "charge_assign",
            agent_id=int(agent.id - 1),
            agent_type=agent.type.name,
            battery=float(agent.battery),
            target=[int(station.x), int(station.y)],
            reason="idle_below_threshold",
        )
        return True

    # ────────────────────────────────────────────────────────────────────
    while not done:
        request_queue = base_env.request_queue
        goal_locations = base_env.goals
        actions = {k: 0 for k in agents}

        # ── 0. Activate any newly-due failures ────────────────────────
        for agent, start_step in failed_starts.items():
            if agent in failed_active or timestep < start_step:
                continue
            failed_active.add(agent)
            assigned_agvs.pop(agent, None)
            assigned_pickers.pop(agent, None)
            assigned_items.pop(agent, None)
            for sid, helper in list(large_group_helpers.items()):
                if helper is agent:
                    large_group_helpers.pop(sid)

        # ── 1. Independent charging: low battery override ─────────────
        for agent in agents:
            if agent in failed_active:
                continue
            if (
                agent.battery < low_battery_threshold
                and agent not in assigned_agvs
                and agent not in assigned_pickers
            ):
                target_dict = (
                    assigned_agvs if agent.type == AgentType.AGV else assigned_pickers
                )
                assign_charge(agent, target_dict)
                if role_tracker is not None and agent.type == AgentType.AGV:
                    role_tracker.on_task_assigned(
                        agent_id=agent.id, mission_type="charging"
                    )

        # ── 2. Synchronized charging: joint prediction before task assign
        if charging_strategy == ChargingStrategy.SYNCHRONIZED and pickers:
            for agv in agvs:
                if agv in failed_active:
                    continue
                if agv in assigned_agvs:
                    continue                 # Already has a mission
                if not request_queue:
                    continue
                paired_picker = agv_to_picker.get(agv)
                if paired_picker is None:
                    continue

                # Estimate steps for the next pick cycle
                # Use nearest requested shelf as proxy
                nearest_item = min(
                    request_queue,
                    key=lambda it: _path_len(env, agv, (it.y, it.x)),
                    default=None,
                )
                if nearest_item is None:
                    continue

                agv_to_shelf = _path_len(env, agv, (nearest_item.y, nearest_item.x))
                # Rough shelf→goal estimate (diagonal of warehouse)
                agv_shelf_to_goal = max(base_env.grid_size) // 2
                agv_to_charger = _nearest_charger_path_len(env, agv)
                picker_to_shelf = _path_len(env, paired_picker, (nearest_item.y, nearest_item.x))
                picker_to_charger = _nearest_charger_path_len(env, paired_picker)

                charge_decision = coordinator.should_charge_before_task(
                    agv_id=agv.id,
                    agv_battery=agv.battery,
                    picker_battery=paired_picker.battery,
                    agv_steps_to_shelf=agv_to_shelf,
                    agv_steps_shelf_to_goal=agv_shelf_to_goal,
                    agv_steps_to_charger=agv_to_charger,
                    picker_steps_to_shelf=picker_to_shelf,
                    picker_steps_to_charger=picker_to_charger,
                )

                if charge_decision["agv"] and agv not in assigned_agvs:
                    assign_charge(agv, assigned_agvs)
                if charge_decision["picker"] and paired_picker not in assigned_pickers:
                    assign_charge(paired_picker, assigned_pickers)

        # ── 3. Task assignment (pluggable planner) ───────────────
        carried_shelf_ids = {a.carrying_shelf.id for a in agents if a.carrying_shelf}
        if request_queue:
            available_agvs = [
                a for a in agvs
                if not a.busy
                and not a.carrying_shelf
                and a not in assigned_agvs
                and a not in failed_active
                and a.battery >= low_battery_threshold
            ]
            free_pickers = [
                p for p in pickers
                if not p.busy
                and not p.carrying_shelf
                and p not in assigned_pickers
                and p not in failed_active
                and p.battery >= low_battery_threshold
            ]

            ctx = AssignmentContext(
                request_queue=request_queue,
                agvs=agvs,
                pickers=pickers,
                available_agvs=available_agvs,
                free_pickers=free_pickers,
                assigned_agvs=assigned_agvs,
                assigned_pickers=assigned_pickers,
                assigned_items=assigned_items,
                carried_shelf_ids=carried_shelf_ids,
                failed_active=failed_active,
                low_battery_threshold=low_battery_threshold,
                timestep=timestep,
                location_id=location_id,
                log_behavior=log_behavior,
            )
            commitments = (assignment_fn or greedy_assignment)(env, ctx)

            for com in commitments:
                if isinstance(com, ChargeDirective):
                    c_agent = com.agent
                    if (
                        c_agent in assigned_agvs
                        or c_agent in assigned_pickers
                        or c_agent in failed_active
                        or c_agent.busy
                    ):
                        continue
                    c_dict = (
                        assigned_agvs if c_agent.type == AgentType.AGV
                        else assigned_pickers
                    )
                    if assign_charge(c_agent, c_dict):
                        if role_tracker is not None and c_agent.type == AgentType.AGV:
                            role_tracker.on_task_assigned(
                                agent_id=c_agent.id, mission_type="charging"
                            )
                    continue

                item = com.item
                item_loc_id = com.item_loc_id
                req_agvs = com.req_agvs
                req_pickers = com.req_pickers
                chosen_agvs = com.agvs
                chosen_pickers = com.pickers

                # Defensive re-validation: a planner may commit from a cached
                # plan while this step's overrides (charging, failures) or an
                # earlier commitment already claimed an agent or the shelf.
                members_free = (
                    all(a in available_agvs and a not in assigned_agvs
                        for a in chosen_agvs)
                    and all(p in free_pickers and p not in assigned_pickers
                            for p in chosen_pickers)
                )
                if (
                    not members_free
                    or len(set(id(a) for a in chosen_agvs)) != len(chosen_agvs)
                    or len(set(id(p) for p in chosen_pickers)) != len(chosen_pickers)
                    or item.id in set(assigned_items.values())
                    or item.id in carried_shelf_ids
                ):
                    log_behavior(
                        "planner_commit_dropped",
                        shelf_id=int(item.id),
                        package_type=item.package_type.name,
                        reason="stale_commitment",
                    )
                    continue

                for agv in chosen_agvs:
                    assigned_agvs[agv] = Mission(
                        MissionType.PICKING, item_loc_id, item.x, item.y, timestep
                    )
                    assigned_items[agv] = item.id

                for picker in chosen_pickers:
                    assigned_pickers[picker] = Mission(
                        MissionType.PICKING, item_loc_id, item.x, item.y, timestep
                    )
                    assigned_items[picker] = item.id

                if req_agvs >= 2 and len(chosen_agvs) >= 2:
                    lead_agv = min(chosen_agvs, key=lambda a: a.id)
                    helper = [a for a in chosen_agvs if a is not lead_agv][0]
                    large_group_helpers[item.id] = helper
                else:
                    lead_agv = chosen_agvs[0] if chosen_agvs else None

                if lead_agv is not None:
                    replan_ctrl.on_mission_assigned(
                        agent_id=lead_agv.id,
                        step=timestep,
                        battery=lead_agv.battery,
                        phase="picking",
                        payload_kg=0.0,
                    )

                if role_tracker is not None and lead_agv is not None:
                    section_idx: Optional[int] = None
                    if pickers and picker_sections:
                        _in_sec = [(item.y, item.x) in sec for sec in picker_sections]
                        if any(_in_sec):
                            section_idx = _in_sec.index(True)
                    role_tracker.on_task_assigned(
                        agent_id=lead_agv.id,
                        mission_type="picking",
                        section_idx=section_idx,
                    )

                if lead_agv is not None and chosen_pickers:
                    agv_to_picker[lead_agv] = chosen_pickers[0]
                    if lead_agv.id in coordinator.pairs:
                        coordinator.pairs[lead_agv.id].picker_id = chosen_pickers[0].id

                log_behavior(
                    "planner_assign",
                    shelf_id=int(item.id),
                    package_type=item.package_type.name,
                    required_agvs=req_agvs,
                    required_pickers=req_pickers,
                    agv_ids=_agent_ids(chosen_agvs),
                    picker_ids=_agent_ids(chosen_pickers),
                    reason=com.reason,
                )
                _notify(
                    "coalition_committed",
                    shelf_id=int(item.id),
                    agvs=chosen_agvs,
                    pickers=chosen_pickers,
                )

        # ── 4. AGV mission transitions ────────────────────────────────
        for agv in agvs:
            if agv in assigned_agvs:
                m = assigned_agvs[agv]
                if agv.x == m.location_x and agv.y == m.location_y:
                    m.at_location = True

            if agv not in assigned_agvs or agv.busy:
                continue

            m = assigned_agvs[agv]

            # PICKING → DELIVERING
            if (m.mission_type == MissionType.PICKING
                    and m.at_location and agv.carrying_shelf):
                closest_goal = _choose_goal(
                    env, agv, goal_locations,
                    _claimed_goal_cells(assigned_agvs, assigned_pickers),
                )
                goal_id = location_id(closest_goal[1], closest_goal[0])
                if goal_id is None:
                    continue
                assigned_agvs.pop(agv)
                assigned_agvs[agv] = Mission(
                    MissionType.DELIVERING,
                    goal_id,
                    closest_goal[0], closest_goal[1],
                    timestep,
                )

                # Record delivering role (completing the pick→deliver→return cycle)
                if role_tracker is not None:
                    role_tracker.on_task_assigned(
                        agent_id=agv.id, mission_type="delivering"
                    )

                # Free LARGE group helper: its job ends once lead carries the shelf
                shelf_id = assigned_items.get(agv)
                if shelf_id is not None and shelf_id in large_group_helpers:
                    helper = large_group_helpers.pop(shelf_id)
                    assigned_agvs.pop(helper, None)
                    assigned_items.pop(helper, None)
                    _notify("mission_complete", agent=helper,
                            mission_type="large_helper_released")

                # Weight revealed: transition replanning tracking to loaded phase
                actual_weight = (
                    agv.carrying_shelf.weight
                    if agv.carrying_shelf is not None
                    else energy_model.pkg_weight_mu
                )
                payload_for_tracking = (
                    actual_weight if use_stage2_recheck
                    else energy_model.pkg_weight_mu
                )
                replan_ctrl.on_phase_transition(
                    agent_id=agv.id,
                    step=timestep,
                    battery=agv.battery,
                    new_phase="delivering",
                    payload_kg=payload_for_tracking,
                )

                # ── Stage 2 recheck (RQ3b): re-evaluate with revealed weight ──
                if use_stage2_recheck and replanning_policy != ReplanningPolicy.NO_REPLANNING:
                    steps_to_goal = _path_len(env, agv, (closest_goal[1], closest_goal[0]))
                    steps_to_charger = _nearest_charger_path_len(env, agv)
                    stage2_feasible = replan_ctrl.check_feasibility_delivering(
                        battery=agv.battery,
                        payload_kg=actual_weight,
                        steps_to_goal=steps_to_goal,
                        steps_to_charger=steps_to_charger,
                        low_battery_threshold=low_battery_threshold,
                    )
                    if not stage2_feasible:
                        # Stage 2 says abort: record and override to charging
                        replan_ctrl.record_decision(
                            agent_id=agv.id,
                            step=timestep,
                            battery_actual=agv.battery,
                            decision="abort_charge",
                            mission_phase="delivering",
                            trigger="weight_revealed",
                        )
                        assigned_agvs.pop(agv)
                        assigned_items.pop(agv, None)
                        replan_ctrl.on_mission_complete(agv.id)
                        assign_charge(agv, assigned_agvs)
                        _notify("abort_charge", agent=agv, phase="delivering",
                                trigger="weight_revealed")
                        if role_tracker is not None:
                            role_tracker.on_task_assigned(
                                agent_id=agv.id, mission_type="charging"
                            )

            # DELIVERING → (RETURNING | None)
            # After process_shelf_deliveries drops the outbound shelf at the
            # goal, the AGV has no carrying_shelf. When return_shelves is True
            # the AGV stays at the goal, picks up the inbound shelf via
            # TOGGLE_LOAD, and navigates back to the shelf's home slot. When
            # return_shelves is False the mission ends here and the AGV is
            # reassigned to a new outbound pick, matching MARL-trained policy
            # behaviour which does not run a restock cycle.
            m = assigned_agvs.get(agv)
            if m and m.mission_type == MissionType.DELIVERING and m.at_location:
                if not return_shelves and not agv.carrying_shelf:
                    assigned_agvs.pop(agv)
                    assigned_items.pop(agv, None)
                    replan_ctrl.on_mission_complete(agv.id)
                    log_behavior(
                        "mission_complete",
                        agent_id=int(agv.id - 1),
                        agent_type=agv.type.name,
                        mission_type="delivering_no_return",
                    )
                    _notify("mission_complete", agent=agv,
                            mission_type="delivering_no_return")
                elif agv.carrying_shelf and agv.carrying_shelf.direction == PackageDirection.IN:
                    # Inbound shelf loaded; navigate to its home slot.
                    shelf = agv.carrying_shelf
                    home_yx = (shelf.home_pos[1], shelf.home_pos[0])  # (row, col)
                    home_id = location_id(home_yx[0], home_yx[1])
                    if home_id is not None:
                        assigned_agvs.pop(agv)
                        assigned_agvs[agv] = Mission(
                            MissionType.RETURNING,
                            home_id,
                            shelf.home_pos[0], shelf.home_pos[1],
                            timestep,
                        )
                        if role_tracker is not None:
                            role_tracker.on_task_assigned(
                                agent_id=agv.id, mission_type="returning"
                            )
                        log_behavior(
                            "mission_transition",
                            agent_id=int(agv.id - 1),
                            agent_type=agv.type.name,
                            mission_type="returning",
                            shelf_id=int(shelf.id),
                            package_type=shelf.package_type.name,
                        )
                # else: AGV is at goal without inbound shelf yet (just delivered,
                # shelf dropped by env) — stay in DELIVERING so attribute_macro_actions
                # keeps issuing TOGGLE_LOAD to pick up the inbound shelf.
                elif not agv.carrying_shelf and not agv.busy:
                    # Liveness guard: the dock pickup can only happen while an
                    # inbound shelf sits on this goal cell. If it is gone (or
                    # was never queued for restock) the AGV would squat on the
                    # dock forever, blocking every later delivery to this goal.
                    cell_shelf_id = base_env.grid[
                        CollisionLayers.SHELVES, agv.y, agv.x
                    ]
                    inbound_ids = {s.id for s in base_env.inbound_queue}
                    if not cell_shelf_id or cell_shelf_id not in inbound_ids:
                        assigned_agvs.pop(agv)
                        assigned_items.pop(agv, None)
                        replan_ctrl.on_mission_complete(agv.id)
                        log_behavior(
                            "mission_complete",
                            agent_id=int(agv.id - 1),
                            agent_type=agv.type.name,
                            mission_type="delivering_dock_released",
                        )
                        _notify("mission_complete", agent=agv,
                                mission_type="delivering_dock_released")

            # RETURNING → None
            m = assigned_agvs.get(agv)
            if m and (m.mission_type == MissionType.RETURNING
                      and m.at_location and not agv.carrying_shelf):
                assigned_agvs.pop(agv)
                assigned_items.pop(agv, None)
                replan_ctrl.on_mission_complete(agv.id)
                log_behavior(
                    "mission_complete",
                    agent_id=int(agv.id - 1),
                    agent_type=agv.type.name,
                    mission_type="returning",
                )
                _notify("mission_complete", agent=agv, mission_type="returning")

        # ── 5. Picker mission transitions and compatibility assignment ─
        # Assist-complete release: a picker still heading to (or waiting at) a
        # PICKING cell whose shelf is already being carried away has nothing
        # left to do there — keeping the stale mission wedges it against
        # oncoming traffic and starves every picker-dependent task.
        for picker in list(assigned_pickers):
            m = assigned_pickers[picker]
            if m.mission_type != MissionType.PICKING or picker.carrying_shelf:
                continue
            s_id = assigned_items.get(picker)
            if s_id is None:
                continue
            if any(
                a.carrying_shelf is not None and a.carrying_shelf.id == s_id
                for a in agvs
            ):
                assigned_pickers.pop(picker)
                assigned_items.pop(picker, None)
                picker.busy = False
                picker.path = []
                log_behavior(
                    "picker_assist_complete",
                    agent_id=int(picker.id - 1),
                    shelf_id=int(s_id),
                )
                _notify("mission_complete", agent=picker,
                        mission_type="assist_complete")

        for picker in pickers:
            if picker in assigned_pickers:
                m = assigned_pickers[picker]
                if picker.x == m.location_x and picker.y == m.location_y:
                    m.at_location = True

            if picker not in assigned_pickers or picker.busy:
                continue

            m = assigned_pickers[picker]
            if (m.mission_type == MissionType.PICKING
                    and m.at_location and picker.carrying_shelf):
                closest_goal = _choose_goal(
                    env, picker, goal_locations,
                    _claimed_goal_cells(assigned_agvs, assigned_pickers),
                )
                goal_id = location_id(closest_goal[1], closest_goal[0])
                if goal_id is None:
                    continue
                assigned_pickers.pop(picker)
                assigned_pickers[picker] = Mission(
                    MissionType.DELIVERING,
                    goal_id,
                    closest_goal[0], closest_goal[1],
                    timestep,
                )
                log_behavior(
                    "mission_transition",
                    agent_id=int(picker.id - 1),
                    agent_type=picker.type.name,
                    mission_type="delivering",
                    shelf_id=int(picker.carrying_shelf.id),
                    package_type=picker.carrying_shelf.package_type.name,
                )

            m = assigned_pickers.get(picker)
            if m and m.mission_type == MissionType.DELIVERING and m.at_location:
                if not return_shelves and not picker.carrying_shelf:
                    assigned_pickers.pop(picker)
                    assigned_items.pop(picker, None)
                    log_behavior(
                        "mission_complete",
                        agent_id=int(picker.id - 1),
                        agent_type=picker.type.name,
                        mission_type="delivering_no_return",
                    )
                    _notify("mission_complete", agent=picker,
                            mission_type="delivering_no_return")
                elif picker.carrying_shelf and picker.carrying_shelf.direction == PackageDirection.IN:
                    shelf = picker.carrying_shelf
                    home_yx = (shelf.home_pos[1], shelf.home_pos[0])
                    home_id = location_id(home_yx[0], home_yx[1])
                    if home_id is not None:
                        assigned_pickers.pop(picker)
                        assigned_pickers[picker] = Mission(
                            MissionType.RETURNING,
                            home_id,
                            shelf.home_pos[0], shelf.home_pos[1],
                            timestep,
                        )
                        log_behavior(
                            "mission_transition",
                            agent_id=int(picker.id - 1),
                            agent_type=picker.type.name,
                            mission_type="returning",
                            shelf_id=int(shelf.id),
                            package_type=shelf.package_type.name,
                        )

            m = assigned_pickers.get(picker)
            if m and (m.mission_type == MissionType.RETURNING
                      and m.at_location and not picker.carrying_shelf):
                assigned_pickers.pop(picker)
                assigned_items.pop(picker, None)
                log_behavior(
                    "mission_complete",
                    agent_id=int(picker.id - 1),
                    agent_type=picker.type.name,
                    mission_type="returning",
                )
                _notify("mission_complete", agent=picker, mission_type="returning")

        for agv, mission in assigned_agvs.items():
            if mission.mission_type not in (MissionType.PICKING, MissionType.RETURNING):
                continue
            if not picker_sections:
                continue
            s_id = assigned_items.get(agv)
            if s_id is not None and s_id in solo_shelf_ids:
                continue  # SOLO task needs no picker
            if s_id is not None and any(
                assigned_items.get(p) == s_id for p in assigned_pickers
            ):
                continue
            loc = (mission.location_y, mission.location_x)
            in_section = [loc in sec for sec in picker_sections]
            if not any(in_section):
                continue
            relevant_picker = pickers[in_section.index(True)]
            if relevant_picker in failed_active:
                continue
            if relevant_picker not in assigned_pickers:
                assigned_pickers[relevant_picker] = Mission(
                    MissionType.PICKING,
                    mission.location_id,
                    mission.location_x, mission.location_y,
                    timestep,
                )
                assigned_items[relevant_picker] = s_id

        # Ensure LARGE tasks have a second picker assigned
        seen_large_locs: set = set()
        for agv, mission in list(assigned_agvs.items()):
            if mission.mission_type not in (MissionType.PICKING, MissionType.RETURNING):
                continue
            s_id = assigned_items.get(agv)
            if s_id is None or s_id not in large_group_helpers:
                continue
            loc_key = (mission.location_x, mission.location_y)
            if loc_key in seen_large_locs:
                continue
            seen_large_locs.add(loc_key)
            already = sum(
                1 for pm in assigned_pickers.values()
                if pm.location_x == mission.location_x
                and pm.location_y == mission.location_y
                and pm.mission_type != MissionType.CHARGING
            )
            if already >= 2:
                continue
            avail = [p for p in pickers if p not in assigned_pickers and p not in failed_active]
            if not avail:
                continue
            dists = [_path_len(env, p, (mission.location_y, mission.location_x)) for p in avail]
            extra = avail[int(np.argmin(dists))]
            assigned_pickers[extra] = Mission(
                MissionType.PICKING, mission.location_id,
                mission.location_x, mission.location_y, timestep,
            )
            assigned_items[extra] = s_id

        # Picker reached destination → clear mission
        pickers_to_remove = []
        for picker in pickers:
            if picker in assigned_pickers:
                m = assigned_pickers[picker]
                if picker.x == m.location_x and picker.y == m.location_y:
                    if m.mission_type != MissionType.CHARGING:
                        assigned_pickers[picker].at_location = True
                        # LARGE shelf: keep picker waiting at the shelf until
                        # the atomic load fires (requires 2 AGVs + 2 pickers
                        # simultaneously). Clear only once the shelf is gone.
                        shelf_id = base_env.grid[CollisionLayers.SHELVES, picker.y, picker.x]
                        if shelf_id:
                            shelf = base_env.shelfs[shelf_id - 1]
                            if getattr(shelf, "required_pickers", 1) > 0:
                                # Liveness guard: only keep waiting while the
                                # cooperative load can still happen — i.e. the
                                # shelf is picker-led, or some AGV still owns
                                # or carries the picker's task. Otherwise the
                                # picker is orphaned (e.g. bound to assist a
                                # RETURNING shelf whose AGV already finished)
                                # and would block all future coalitions.
                                s_id = assigned_items.get(picker)
                                picker_led = getattr(shelf, "required_agvs", 1) == 0
                                partner_active = s_id is not None and (
                                    any(assigned_items.get(a) == s_id for a in agvs)
                                    or any(
                                        a.carrying_shelf is not None
                                        and a.carrying_shelf.id == s_id
                                        for a in agvs
                                    )
                                )
                                if picker_led or partner_active:
                                    continue
                                log_behavior(
                                    "picker_orphan_release",
                                    agent_id=int(picker.id - 1),
                                    shelf_id=int(s_id) if s_id is not None else -1,
                                )
                        pickers_to_remove.append(picker)
        for p in pickers_to_remove:
            assigned_pickers.pop(p)
            assigned_items.pop(p, None)

        # ── 5c. Liveness circuit breaker for traffic standoffs ────────
        for agent in agents:
            mission = assigned_agvs.get(agent) or assigned_pickers.get(agent)
            if (
                mission is None
                or mission.at_location
                or mission.mission_type == MissionType.CHARGING
                or agent in failed_active
            ):
                stagnation.pop(agent, None)
                continue
            sig = (agent.x, agent.y, id(mission))
            prev = stagnation.get(agent)
            count = prev[1] + 1 if prev is not None and prev[0] == sig else 0
            stagnation[agent] = (sig, count)
            if count < STUCK_RELEASE_STEPS:
                continue
            stagnation[agent] = (sig, 0)

            if (
                mission.mission_type == MissionType.DELIVERING
                and agent.type == AgentType.AGV
                and agent.carrying_shelf
            ):
                # Re-route the delivery to an alternative dock that is neither
                # claimed nor physically occupied, dissolving head-on locks in
                # single-lane dock corridors.
                occupied_cells = {(a.x, a.y) for a in agents if a is not agent}
                claimed = _claimed_goal_cells(assigned_agvs, assigned_pickers)
                ranked = sorted(
                    goal_locations,
                    key=lambda g: _path_len(env, agent, (g[1], g[0])),
                )
                new_goal = next(
                    (
                        g for g in ranked
                        if (g[0], g[1]) != (mission.location_x, mission.location_y)
                        and (g[0], g[1]) not in claimed
                        and (g[0], g[1]) not in occupied_cells
                        # only retarget when the alternative dock is actually
                        # reachable right now, otherwise dock ping-pong thrash
                        and _path_len(env, agent, (g[1], g[0])) < int(1e5)
                    ),
                    None,
                )
                if new_goal is not None:
                    gid = location_id(new_goal[1], new_goal[0])
                    if gid is not None:
                        assigned_agvs.pop(agent)
                        assigned_agvs[agent] = Mission(
                            MissionType.DELIVERING,
                            gid,
                            new_goal[0], new_goal[1],
                            timestep,
                        )
                        agent.busy = False
                        agent.path = []
                        log_behavior(
                            "liveness_retarget_goal",
                            agent_id=int(agent.id - 1),
                            agent_type=agent.type.name,
                            new_target=[int(new_goal[0]), int(new_goal[1])],
                        )
                        continue

            # Generic escape: clear the macro so the path is recomputed from
            # scratch against current agent positions.
            agent.busy = False
            agent.path = []
            log_behavior(
                "liveness_repath",
                agent_id=int(agent.id - 1),
                agent_type=agent.type.name,
                mission_type=mission.mission_type.name,
            )

        # ── 6. Map missions → actions ─────────────────────────────────
        agvs_done = []
        for agv, mission in assigned_agvs.items():
            if mission.mission_type == MissionType.CHARGING:
                at_station = (
                    agv.x == mission.location_x
                    and agv.y == mission.location_y
                )
                if at_station and agv.battery < _BATTERY_FULL:
                    actions[agv] = 0
                    agv.path = []
                    agv.charging = True
                else:
                    actions[agv] = mission.location_id if not agv.busy else 0
                if agv.battery >= _BATTERY_FULL:
                    agvs_done.append(agv)
            else:
                actions[agv] = mission.location_id if not agv.busy else 0
        for agv in agvs_done:
            assigned_agvs.pop(agv)
            log_behavior(
                "charge_complete",
                agent_id=int(agv.id - 1),
                agent_type=agv.type.name,
                battery=float(agv.battery),
            )
            _notify("charge_complete", agent=agv)

        pickers_done = []
        for picker, mission in assigned_pickers.items():
            if mission.mission_type == MissionType.CHARGING:
                at_station = (
                    picker.x == mission.location_x
                    and picker.y == mission.location_y
                )
                if at_station and picker.battery < _BATTERY_FULL:
                    actions[picker] = 0
                    picker.path = []
                    picker.charging = True
                else:
                    actions[picker] = mission.location_id
                if picker.battery >= _BATTERY_FULL:
                    pickers_done.append(picker)
            else:
                actions[picker] = mission.location_id
        for p in pickers_done:
            assigned_pickers.pop(p)
            log_behavior(
                "charge_complete",
                agent_id=int(p.id - 1),
                agent_type=p.type.name,
                battery=float(p.battery),
            )
            _notify("charge_complete", agent=p)

        # ── 7. Coupling metrics: record AGV blocked due to picker ──────
        for agv in agvs:
            paired_picker = agv_to_picker.get(agv)
            if paired_picker is None:
                continue
            # AGV is "blocked by picker" if it has a PICKING mission at location
            # but picker is absent (charging or en route to charger)
            agv_waiting_for_picker = (
                agv in assigned_agvs
                and assigned_agvs[agv].mission_type == MissionType.PICKING
                and assigned_agvs[agv].at_location
                and not agv.carrying_shelf
            )
            picker_unavailable = (
                paired_picker in assigned_pickers
                and assigned_pickers[paired_picker].mission_type == MissionType.CHARGING
            ) or paired_picker.battery == 0

            coordinator.record_step(
                agv_id=agv.id,
                agv_battery=agv.battery,
                picker_battery=paired_picker.battery,
                agv_is_blocked=agv_waiting_for_picker and picker_unavailable,
                picker_is_blocked=False,
                step=timestep,
                deliveries=total_deliveries,
            )

        # ── 7b. RQ3: Mid-mission replanning check ─────────────────────
        # For each AGV with an active PICKING or DELIVERING mission,
        # check if the current replanning policy fires a replan trigger.
        # If triggered: run feasibility check; abort to charge if infeasible.
        if replanning_policy != ReplanningPolicy.NO_REPLANNING:
            agvs_to_abort = []
            for agv in agvs:
                m = assigned_agvs.get(agv)
                if m is None or m.mission_type not in (
                    MissionType.PICKING, MissionType.DELIVERING
                ):
                    continue
                # Safety violation: battery hit 0 mid-task
                if agv.battery == 0:
                    replan_ctrl.record_safety_violation()

                if not replan_ctrl.should_replan(agv.id, timestep, agv.battery):
                    continue

                # Determine feasibility from current position
                phase = (
                    "picking" if m.mission_type == MissionType.PICKING
                    else "delivering"
                )
                steps_to_charger = _nearest_charger_path_len(base_env, agv)

                if m.mission_type == MissionType.PICKING:
                    # Remaining: travel to shelf + shelf→goal
                    steps_to_shelf = _path_len(base_env, agv, (m.location_y, m.location_x))
                    steps_shelf_to_goal = max(base_env.grid_size) // 2
                    feasible = replan_ctrl.check_feasibility_picking(
                        battery=agv.battery,
                        steps_to_shelf=steps_to_shelf,
                        steps_shelf_to_goal=steps_shelf_to_goal,
                        steps_to_charger=steps_to_charger,
                        alpha=alpha,
                        low_battery_threshold=low_battery_threshold,
                    )
                else:  # DELIVERING
                    steps_to_goal = _path_len(base_env, agv, (m.location_y, m.location_x))
                    payload_kg = (
                        agv.carrying_shelf.weight
                        if agv.carrying_shelf is not None
                        else energy_model.pkg_weight_mu
                    )
                    feasible = replan_ctrl.check_feasibility_delivering(
                        battery=agv.battery,
                        payload_kg=payload_kg,
                        steps_to_goal=steps_to_goal,
                        steps_to_charger=steps_to_charger,
                        low_battery_threshold=low_battery_threshold,
                    )

                decision = "continue" if feasible else "abort_charge"
                replan_ctrl.record_decision(
                    agent_id=agv.id,
                    step=timestep,
                    battery_actual=agv.battery,
                    decision=decision,
                    mission_phase=phase,
                )

                if decision == "abort_charge":
                    agvs_to_abort.append(agv)

            # Execute aborts: drop current mission, assign charging
            for agv in agvs_to_abort:
                assigned_agvs.pop(agv, None)
                assigned_items.pop(agv, None)
                replan_ctrl.on_mission_complete(agv.id)
                assign_charge(agv, assigned_agvs)
                _notify("abort_charge", agent=agv, phase="mid_mission",
                        trigger="replan_feasibility")
                if role_tracker is not None:
                    role_tracker.on_task_assigned(
                        agent_id=agv.id, mission_type="charging"
                    )

        # ── 8. Step ───────────────────────────────────────────────────
        if render:
            runtime_env.unwrapped.render(mode="human")
        if save_gif:
            frame = runtime_env.unwrapped.render(mode="rgb_array")
            frames.append(frame)

        # Force NOOP for every failed agent; tally the audit.
        for agent in failed_active:
            if actions[agent] != 0:
                failure_audit[agent_to_failure_idx[agent]]["all_zero"] = False
            actions[agent] = 0
            failure_audit[agent_to_failure_idx[agent]]["override_count"] += 1

        _, reward, terminated, truncated, info = runtime_env.step(list(actions.values()))
        total_deliveries += info.get("shelf_deliveries", 0)
        for event in info.get("delivery_events", []):
            log_behavior("delivery_event", **event)
            _notify("delivery_event", **event)
        for event in info.get("return_events", []):
            log_behavior("return_event", **event)
        done = all(terminated) or all(truncated)
        episode_returns += np.array(reward, dtype=np.float64)
        global_episode_return += float(np.sum(reward))
        all_infos.append(info)
        timestep += 1

    if save_gif and frames:
        imageio.mimsave(gif_path, frames, fps=10)

    coupling_metrics = coordinator.get_all_metrics()
    replanning_metrics = replan_ctrl.get_metrics()
    role_metrics = role_tracker.get_metrics() if role_tracker is not None else None
    log_behavior(
        "episode_end",
        total_steps=int(timestep),
        total_deliveries=int(total_deliveries),
        termination="terminated_or_truncated",
    )
    if failure_specs:
        log_behavior(
            "failure_audit",
            failures=[
                {
                    "agent_index": idx,
                    "start_step": rec["start_step"],
                    "override_count": rec["override_count"],
                    "all_zero": bool(rec["all_zero"]),
                }
                for idx, rec in failure_audit.items()
            ],
        )
    return (
        all_infos,
        global_episode_return,
        episode_returns,
        coupling_metrics,
        replanning_metrics,
        role_metrics,
        behavior_events,
    )
