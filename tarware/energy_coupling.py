"""
Energy-coupling helpers for heterogeneous picker-AGV teams.

This module defines the charging-strategy enum, AGV and picker energy-cost
model, per-pair coupling metrics, picker-AGV depletion prediction, Stage-2
post-pickup rechecks with revealed payload weight, and the coordinator used by
the heuristic runner to choose independent or synchronized charging windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Charging strategy enum
# ────────────────────────────────────────────────────────────────────────────

class ChargingStrategy(Enum):
    INDEPENDENT = "independent"    # Each agent charges when own battery < threshold
    SYNCHRONIZED = "synchronized"  # Paired agents charge together (joint prediction)


# ────────────────────────────────────────────────────────────────────────────
# Heterogeneous energy model
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class EnergyModel:
    """
    Heterogeneous energy consumption parameters for AGV and Picker.

    AGV model (load + distance driven):
        E_move = base_cost * (1 + weight_factor * payload_kg)   per step
        E_toggle = load_cost                                     per TOGGLE_LOAD

    Picker model (time + action driven):
        E_move = base_cost                                       per step
        E_lift  = lift_cost                                      per lift assist
    """
    # AGV
    agv_move_cost: float = 1.0      # Base energy per forward step
    agv_weight_factor: float = 0.04 # Fractional extra energy per payload kg
    agv_load_cost: float = 2.0      # Energy per TOGGLE_LOAD action

    # Picker
    picker_move_cost: float = 1.0   # Base energy per forward step
    picker_lift_cost: float = 1.0   # Extra energy per lift assist

    # Package weight prior (for pre-assignment planning under uncertainty)
    pkg_weight_mu: float = 5.0      # Prior mean (kg)
    pkg_weight_sigma: float = 2.0   # Prior std  (kg)

    def agv_step_cost(self, payload_kg: float = 0.0) -> float:
        """Expected energy per forward step for AGV with given payload."""
        return self.agv_move_cost * (1.0 + self.agv_weight_factor * payload_kg)

    def agv_path_cost(self, path_length: int, payload_kg: float = 0.0) -> float:
        """Total AGV energy for a path of given step count."""
        return path_length * self.agv_step_cost(payload_kg)

    def picker_path_cost(self, path_length: int, n_lifts: int = 0) -> float:
        """Total Picker energy for a path with n_lifts assist actions."""
        return (path_length * self.picker_move_cost
                + n_lifts * self.picker_lift_cost)

    def agv_path_cost_prior(self, path_length: int) -> Tuple[float, float]:
        """
        (mean, std) of AGV path cost under package weight uncertainty.
        Uses pre-assignment prior N(mu_pkg, sigma_pkg).
        Implements Stage 1 of the two-stage safety framework.
        """
        mu = self.agv_path_cost(path_length, self.pkg_weight_mu)
        # Uncertainty propagated through weight factor
        sigma = path_length * self.agv_move_cost * self.agv_weight_factor * self.pkg_weight_sigma
        return mu, sigma

    def chance_constraint_threshold(self, mu: float, sigma: float,
                                    alpha: float) -> float:
        """
        Conservative energy threshold for alpha-level safety.
        Pr(E_actual <= E_threshold) >= alpha

        Uses normal approximation:  threshold = mu + Phi^{-1}(alpha) * sigma
        """
        from scipy.stats import norm
        return mu + norm.ppf(alpha) * sigma


# ────────────────────────────────────────────────────────────────────────────
# Coupling metrics tracker
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class CouplingMetrics:
    """
    Tracks energy coupling metrics for a picker-AGV pair over one episode.

    Key RQ2 metrics:
      - agv_blocked_steps: steps AGV was idle because picker was depleted / charging
      - picker_blocked_steps: steps picker was idle because AGV was depleted / charging
      - joint_charging_events: times both were sent to charge together
      - independent_charging_events: times one charged while other continued working
      - throughput_loss_fraction: fraction of episode time lost to coupling idle
    """
    episode_length: int = 0
    total_deliveries: int = 0

    # Per-agent charging counts
    agv_charging_events: int = 0
    picker_charging_events: int = 0

    # Coupling-specific
    joint_charging_events: int = 0      # SYNCHRONIZED strategy: both at once
    agv_blocked_steps: int = 0          # AGV idle due to picker unavailability
    picker_blocked_steps: int = 0       # Picker idle due to AGV unavailability

    # Charging window records: (start_step, duration, deliveries_missed)
    charging_windows: List[Tuple[int, int, int]] = field(default_factory=list)

    # Battery trace for analysis
    agv_battery_trace: List[float] = field(default_factory=list)
    picker_battery_trace: List[float] = field(default_factory=list)

    def record_battery(self, agv_bat: float, picker_bat: float) -> None:
        self.agv_battery_trace.append(agv_bat)
        self.picker_battery_trace.append(picker_bat)

    def throughput_loss_fraction(self) -> float:
        """Fraction of episode steps lost to coupling-induced idle."""
        total_idle = self.agv_blocked_steps + self.picker_blocked_steps
        return total_idle / max(self.episode_length, 1)

    def deliveries_per_step(self) -> float:
        return self.total_deliveries / max(self.episode_length, 1)

    def mean_charging_window_duration(self) -> float:
        if not self.charging_windows:
            return 0.0
        return float(np.mean([w[1] for w in self.charging_windows]))

    def summary(self) -> Dict:
        return {
            "episode_length": self.episode_length,
            "total_deliveries": self.total_deliveries,
            "deliveries_per_step": self.deliveries_per_step(),
            "agv_charging_events": self.agv_charging_events,
            "picker_charging_events": self.picker_charging_events,
            "joint_charging_events": self.joint_charging_events,
            "agv_blocked_steps": self.agv_blocked_steps,
            "picker_blocked_steps": self.picker_blocked_steps,
            "throughput_loss_fraction": self.throughput_loss_fraction(),
            "n_charging_windows": len(self.charging_windows),
            "mean_charging_window_duration": self.mean_charging_window_duration(),
        }


# ────────────────────────────────────────────────────────────────────────────
# Picker-AGV pair with joint charging coordination
# ────────────────────────────────────────────────────────────────────────────

class PickerAGVPair:
    """
    Represents a mutualistic picker-AGV pair.

    Tracks:
      - Joint energy state
      - Coupling metrics
      - Two-stage safety thresholds

    The AGV depends on the picker for loading; the picker depends on the AGV
    for meaningful work. When either depletes, both become effectively blocked.
    """

    def __init__(
        self,
        agv_id: int,
        picker_id: int,
        energy_model: Optional[EnergyModel] = None,
        alpha: float = 0.9,
    ):
        self.agv_id = agv_id
        self.picker_id = picker_id
        self.model = energy_model or EnergyModel()
        self.alpha = alpha           # Safety confidence level
        self.metrics = CouplingMetrics()

        self._charging_window_start: Optional[int] = None
        self._deliveries_at_window_start: int = 0

    # ------------------------------------------------------------------
    # Stage 1: Pre-assignment depletion prediction (weight prior)
    # ------------------------------------------------------------------

    def predict_depletion_preassignment(
        self,
        agv_battery: float,
        picker_battery: float,
        agv_steps_to_shelf: int,
        agv_steps_shelf_to_goal: int,
        agv_steps_to_charger: int,
        picker_steps_to_shelf: int,
        picker_steps_to_charger: int,
        low_battery_threshold: float = 10.0,
    ) -> Dict[str, bool]:
        """
        Stage 1: Predict if either agent will deplete before completing the
        next pick-deliver cycle, using the weight prior.

        Returns: {'agv_needs_charge': bool, 'picker_needs_charge': bool,
                  'joint_charge_recommended': bool}
        """
        # AGV: must travel to shelf (empty), then to goal (loaded, weight uncertain)
        agv_empty_cost = self.model.agv_path_cost(agv_steps_to_shelf)
        loaded_mu, loaded_sigma = self.model.agv_path_cost_prior(agv_steps_shelf_to_goal)
        agv_task_cost = (agv_empty_cost
                         + self.model.chance_constraint_threshold(loaded_mu, loaded_sigma, self.alpha)
                         + self.model.agv_load_cost * 2)  # load + unload
        agv_charger_cost = self.model.agv_path_cost(agv_steps_to_charger)
        agv_needs_charge = (agv_battery - agv_task_cost - agv_charger_cost
                            < low_battery_threshold)

        # Picker: travel to shelf + 2 lift assists
        picker_task_cost = self.model.picker_path_cost(picker_steps_to_shelf, n_lifts=2)
        picker_charger_cost = self.model.picker_path_cost(picker_steps_to_charger)
        picker_needs_charge = (picker_battery - picker_task_cost - picker_charger_cost
                               < low_battery_threshold)

        return {
            "agv_needs_charge": agv_needs_charge,
            "picker_needs_charge": picker_needs_charge,
            "joint_charge_recommended": agv_needs_charge or picker_needs_charge,
        }

    # ------------------------------------------------------------------
    # Stage 2: Post-pickup replanning (revealed weight)
    # ------------------------------------------------------------------

    def recheck_after_pickup(
        self,
        agv_battery: float,
        payload_kg: float,      # Revealed after picker scans shelf
        agv_steps_to_goal: int,
        agv_steps_to_charger: int,
        low_battery_threshold: float = 10.0,
    ) -> bool:
        """
        Stage 2: After weight is revealed (picker loads shelf), recheck AGV
        energy feasibility with tight threshold (sigma → 0).

        Returns True if AGV must abort delivery and charge immediately.
        This implements the theoretical Stage 2 from the framework:
          Stage 1 acceptance ⟹ Stage 2 feasibility with probability >= alpha^2.
        """
        agv_task_cost = self.model.agv_path_cost(agv_steps_to_goal, payload_kg)
        agv_task_cost += self.model.agv_load_cost  # unload at goal
        agv_charger_cost = self.model.agv_path_cost(agv_steps_to_charger)
        return (agv_battery - agv_task_cost - agv_charger_cost
                < low_battery_threshold)

    # ------------------------------------------------------------------
    # Charging window tracking
    # ------------------------------------------------------------------

    def start_charging_window(self, step: int, deliveries_so_far: int) -> None:
        self._charging_window_start = step
        self._deliveries_at_window_start = deliveries_so_far

    def end_charging_window(self, step: int, deliveries_so_far: int) -> None:
        if self._charging_window_start is None:
            return
        duration = step - self._charging_window_start
        missed = deliveries_so_far - self._deliveries_at_window_start
        self.metrics.charging_windows.append(
            (self._charging_window_start, duration, missed)
        )
        self._charging_window_start = None


# ────────────────────────────────────────────────────────────────────────────
# Joint charging coordinator
# ────────────────────────────────────────────────────────────────────────────

class JointChargingCoordinator:
    """
    Determines charging decisions for picker-AGV pairs based on strategy.

    INDEPENDENT: Each agent charges when its own battery < threshold.
                 Simple but may leave partner idle mid-task.

    SYNCHRONIZED: Predict joint depletion before accepting a task.
                  If either will deplete before completing the task cycle,
                  send both to charge first. Minimizes mid-task blocking.
    """

    def __init__(
        self,
        pairs: List[PickerAGVPair],
        strategy: ChargingStrategy,
        low_battery_threshold: float = 10.0,
    ):
        self.pairs = {p.agv_id: p for p in pairs}
        self.strategy = strategy
        self.threshold = low_battery_threshold

    def should_charge_before_task(
        self,
        agv_id: int,
        agv_battery: float,
        picker_battery: float,
        agv_steps_to_shelf: int,
        agv_steps_shelf_to_goal: int,
        agv_steps_to_charger: int,
        picker_steps_to_shelf: int,
        picker_steps_to_charger: int,
    ) -> Dict[str, bool]:
        """
        Called before assigning a new pick task to an AGV.

        Returns: {'agv': bool, 'picker': bool}
        """
        if self.strategy == ChargingStrategy.INDEPENDENT:
            return {
                "agv": agv_battery < self.threshold,
                "picker": picker_battery < self.threshold,
            }

        # SYNCHRONIZED: use joint depletion prediction
        pair = self.pairs.get(agv_id)
        if pair is None:
            return {"agv": agv_battery < self.threshold,
                    "picker": picker_battery < self.threshold}

        prediction = pair.predict_depletion_preassignment(
            agv_battery=agv_battery,
            picker_battery=picker_battery,
            agv_steps_to_shelf=agv_steps_to_shelf,
            agv_steps_shelf_to_goal=agv_steps_shelf_to_goal,
            agv_steps_to_charger=agv_steps_to_charger,
            picker_steps_to_shelf=picker_steps_to_shelf,
            picker_steps_to_charger=picker_steps_to_charger,
            low_battery_threshold=self.threshold,
        )

        if prediction["joint_charge_recommended"]:
            # Both charge together → synchronized window
            pair.metrics.joint_charging_events += 1
            return {"agv": True, "picker": True}

        # Emergency fallback: below absolute threshold
        return {
            "agv": agv_battery < self.threshold * 0.5,
            "picker": picker_battery < self.threshold * 0.5,
        }

    def record_step(
        self,
        agv_id: int,
        agv_battery: float,
        picker_battery: float,
        agv_is_blocked: bool,
        picker_is_blocked: bool,
        step: int,
        deliveries: int,
    ) -> None:
        """Update coupling metrics for this timestep."""
        pair = self.pairs.get(agv_id)
        if pair is None:
            return
        pair.metrics.episode_length += 1
        pair.metrics.total_deliveries = deliveries
        pair.metrics.record_battery(agv_battery, picker_battery)
        if agv_is_blocked:
            pair.metrics.agv_blocked_steps += 1
        if picker_is_blocked:
            pair.metrics.picker_blocked_steps += 1

    def get_all_metrics(self) -> Dict[int, Dict]:
        """Return summary metrics for all pairs."""
        return {agv_id: pair.metrics.summary()
                for agv_id, pair in self.pairs.items()}
