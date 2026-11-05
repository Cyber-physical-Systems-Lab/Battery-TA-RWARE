"""
Replanning helpers for the heuristic runner.

The heuristic uses this module to decide when an AGV should keep its current
mission, abort to charge, or re-evaluate feasibility after a shelf weight is
revealed. The implementation stays local to this repository so evaluation does
not depend on an external `tarware` installation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from statistics import NormalDist
from typing import Dict, Optional

from tarware.energy_coupling import EnergyModel


class ReplanningPolicy(Enum):
    """Trigger policy for mid-mission replanning."""

    NO_REPLANNING = "no_replanning"
    FIXED_INTERVAL = "fixed_interval"
    EVENT_TRIGGERED = "event_triggered"


@dataclass
class ReplanningAgentState:
    """Per-agent bookkeeping used by the controller."""

    mission_phase: Optional[str] = None
    payload_kg: float = 0.0
    assigned_step: Optional[int] = None
    assigned_battery: Optional[float] = None
    next_replan_step: Optional[int] = None
    last_replan_step: Optional[int] = None
    last_decision: Optional[str] = None
    last_decision_step: Optional[int] = None


@dataclass
class ReplanningMetrics:
    """Episode-level replanning summary returned by the heuristic runner."""

    policy: str
    replan_interval: int
    deviation_threshold: float
    use_stage2_recheck: bool
    mission_assignments: int = 0
    phase_transitions: int = 0
    replanning_checks: int = 0
    replanning_triggers: int = 0
    feasibility_checks_picking: int = 0
    feasibility_checks_delivering: int = 0
    continue_decisions: int = 0
    abort_charge_decisions: int = 0
    safety_violations: int = 0
    completed_missions: int = 0
    agents: Dict[int, ReplanningAgentState] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        return {
            "policy": self.policy,
            "replan_interval": self.replan_interval,
            "deviation_threshold": self.deviation_threshold,
            "use_stage2_recheck": self.use_stage2_recheck,
            "mission_assignments": self.mission_assignments,
            "phase_transitions": self.phase_transitions,
            "replanning_checks": self.replanning_checks,
            "replanning_triggers": self.replanning_triggers,
            "feasibility_checks_picking": self.feasibility_checks_picking,
            "feasibility_checks_delivering": self.feasibility_checks_delivering,
            "continue_decisions": self.continue_decisions,
            "abort_charge_decisions": self.abort_charge_decisions,
            "safety_violations": self.safety_violations,
            "completed_missions": self.completed_missions,
        }


class ReplanningController:
    """Local replanning controller used by `tarware.heuristic`."""

    def __init__(
        self,
        policy: ReplanningPolicy,
        energy_model: Optional[EnergyModel] = None,
        replan_interval: int = 5,
        deviation_threshold: float = 8.0,
        use_stage2_recheck: bool = True,
    ) -> None:
        self.policy = policy
        self.energy_model = energy_model or EnergyModel()
        self.replan_interval = int(replan_interval)
        self.deviation_threshold = float(deviation_threshold)
        self.use_stage2_recheck = bool(use_stage2_recheck)

        self._states: Dict[int, ReplanningAgentState] = {}
        self._mission_assignments = 0
        self._phase_transitions = 0
        self._replanning_checks = 0
        self._replanning_triggers = 0
        self._feasibility_checks_picking = 0
        self._feasibility_checks_delivering = 0
        self._continue_decisions = 0
        self._abort_charge_decisions = 0
        self._safety_violations = 0
        self._completed_missions = 0

    def _state(self, agent_id: int) -> ReplanningAgentState:
        state = self._states.get(agent_id)
        if state is None:
            state = ReplanningAgentState()
            self._states[agent_id] = state
        return state

    def on_mission_assigned(
        self,
        agent_id: int,
        step: int,
        battery: float,
        phase: str,
        payload_kg: float,
    ) -> None:
        state = self._state(agent_id)
        state.mission_phase = phase
        state.payload_kg = float(payload_kg)
        state.assigned_step = int(step)
        state.assigned_battery = float(battery)
        state.next_replan_step = int(step) + max(self.replan_interval, 1)
        state.last_replan_step = None
        state.last_decision = None
        state.last_decision_step = None
        self._mission_assignments += 1

    def on_phase_transition(
        self,
        agent_id: int,
        step: int,
        battery: float,
        new_phase: str,
        payload_kg: float,
    ) -> None:
        state = self._state(agent_id)
        state.mission_phase = new_phase
        state.payload_kg = float(payload_kg)
        state.assigned_step = int(step)
        state.assigned_battery = float(battery)
        state.next_replan_step = int(step) + max(self.replan_interval, 1)
        self._phase_transitions += 1

    def should_replan(self, agent_id: int, step: int, battery: float) -> bool:
        self._replanning_checks += 1
        if self.policy == ReplanningPolicy.NO_REPLANNING:
            return False

        state = self._state(agent_id)
        if self.policy == ReplanningPolicy.FIXED_INTERVAL:
            next_step = state.next_replan_step
            if next_step is None:
                state.next_replan_step = int(step) + max(self.replan_interval, 1)
                return False
            if step >= next_step:
                state.next_replan_step = int(step) + max(self.replan_interval, 1)
                state.last_replan_step = int(step)
                self._replanning_triggers += 1
                return True
            return False

        baseline = state.assigned_battery if state.assigned_battery is not None else float(battery)
        battery_gap = baseline - float(battery)
        interval_due = state.next_replan_step is not None and step >= state.next_replan_step
        if battery_gap >= self.deviation_threshold or interval_due:
            state.next_replan_step = int(step) + max(self.replan_interval, 1)
            state.last_replan_step = int(step)
            self._replanning_triggers += 1
            return True
        return False

    def record_safety_violation(self) -> None:
        self._safety_violations += 1

    def check_feasibility_picking(
        self,
        battery: float,
        steps_to_shelf: int,
        steps_shelf_to_goal: int,
        steps_to_charger: int,
        alpha: float,
        low_battery_threshold: float = 10.0,
    ) -> bool:
        self._feasibility_checks_picking += 1
        mu, sigma = self.energy_model.agv_path_cost_prior(int(steps_shelf_to_goal))
        z = NormalDist().inv_cdf(float(alpha))
        conservative_loaded_cost = mu + z * sigma
        task_cost = self.energy_model.agv_path_cost(int(steps_to_shelf))
        task_cost += conservative_loaded_cost
        task_cost += 2.0 * self.energy_model.agv_load_cost
        charger_cost = self.energy_model.agv_path_cost(int(steps_to_charger))
        return (float(battery) - task_cost - charger_cost) >= float(low_battery_threshold)

    def check_feasibility_delivering(
        self,
        battery: float,
        payload_kg: float,
        steps_to_goal: int,
        steps_to_charger: int,
        low_battery_threshold: float = 10.0,
    ) -> bool:
        self._feasibility_checks_delivering += 1
        task_cost = self.energy_model.agv_path_cost(int(steps_to_goal), float(payload_kg))
        task_cost += self.energy_model.agv_load_cost
        charger_cost = self.energy_model.agv_path_cost(int(steps_to_charger))
        return (float(battery) - task_cost - charger_cost) >= float(low_battery_threshold)

    def record_decision(
        self,
        agent_id: int,
        step: int,
        battery_actual: float,
        decision: str,
        mission_phase: str,
        trigger: Optional[str] = None,
    ) -> None:
        state = self._state(agent_id)
        state.last_decision = decision
        state.last_decision_step = int(step)
        state.mission_phase = mission_phase
        if decision == "abort_charge":
            self._abort_charge_decisions += 1
        else:
            self._continue_decisions += 1

    def on_mission_complete(self, agent_id: int) -> None:
        self._completed_missions += 1
        state = self._state(agent_id)
        state.mission_phase = None
        state.payload_kg = 0.0
        state.assigned_step = None
        state.assigned_battery = None
        state.next_replan_step = None

    def get_metrics(self) -> ReplanningMetrics:
        return ReplanningMetrics(
            policy=self.policy.value,
            replan_interval=self.replan_interval,
            deviation_threshold=self.deviation_threshold,
            use_stage2_recheck=self.use_stage2_recheck,
            mission_assignments=self._mission_assignments,
            phase_transitions=self._phase_transitions,
            replanning_checks=self._replanning_checks,
            replanning_triggers=self._replanning_triggers,
            feasibility_checks_picking=self._feasibility_checks_picking,
            feasibility_checks_delivering=self._feasibility_checks_delivering,
            continue_decisions=self._continue_decisions,
            abort_charge_decisions=self._abort_charge_decisions,
            safety_violations=self._safety_violations,
            completed_missions=self._completed_missions,
            agents={aid: state for aid, state in self._states.items()},
        )
