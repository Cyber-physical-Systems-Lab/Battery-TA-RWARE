from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from battery_tarware.warehouse import (
    _AGV_WEIGHT_FACTOR,
    _BATTERY_CONSUMPTION_LOAD,
    _BATTERY_CONSUMPTION_MOVE,
    _PICKER_LIFT_ENERGY,
    _PKG_WEIGHT_MU,
)


class ChargingStrategy(Enum):
    INDEPENDENT = "independent"
    SYNCHRONIZED = "synchronized"


@dataclass
class CouplingMetrics:
    joint_charge_decisions: int = 0
    agv_charge_decisions: int = 0
    picker_charge_decisions: int = 0


class EnergyModel:
    def __init__(
        self,
        move_cost: float = _BATTERY_CONSUMPTION_MOVE,
        agv_weight_factor: float = _AGV_WEIGHT_FACTOR,
        agv_load_cost: float = _BATTERY_CONSUMPTION_LOAD,
        picker_lift_energy: float = _PICKER_LIFT_ENERGY,
        pkg_weight_mu: float = _PKG_WEIGHT_MU,
    ):
        self.move_cost = float(move_cost)
        self.agv_weight_factor = float(agv_weight_factor)
        self.agv_load_cost = float(agv_load_cost)
        self.picker_lift_energy = float(picker_lift_energy)
        self.pkg_weight_mu = float(pkg_weight_mu)

    def agv_path_cost(self, steps: int, payload_kg: float = 0.0) -> float:
        payload = max(0.0, float(payload_kg))
        return float(steps) * self.move_cost * (1.0 + self.agv_weight_factor * payload)

    def picker_path_cost(self, steps: int, n_lifts: int = 0) -> float:
        lifts = max(0, int(n_lifts))
        return float(steps) * self.move_cost + lifts * self.picker_lift_energy


@dataclass
class PickerAGVPair:
    agv_id: int
    picker_id: int
    energy_model: EnergyModel
    alpha: float = 0.9


class JointChargingCoordinator:
    def __init__(
        self,
        pairs: List[PickerAGVPair],
        strategy: ChargingStrategy = ChargingStrategy.INDEPENDENT,
        low_battery_threshold: float = 10.0,
    ):
        self.pairs: Dict[int, PickerAGVPair] = {p.agv_id: p for p in pairs}
        self.strategy = strategy
        self.low_battery_threshold = float(low_battery_threshold)
        self.metrics = CouplingMetrics()

    def should_charge_before_task(
        self,
        *,
        agv_id: int,
        agv_battery: float,
        picker_battery: float,
        agv_steps_to_shelf: int,
        agv_steps_shelf_to_goal: int,
        agv_steps_to_charger: int,
        picker_steps_to_shelf: int,
        picker_steps_to_charger: int,
    ) -> Dict[str, bool]:
        pair = self.pairs.get(agv_id)
        model = pair.energy_model if pair else EnergyModel()

        agv_cycle_cost = (
            model.agv_path_cost(agv_steps_to_shelf, payload_kg=0.0)
            + model.agv_load_cost
            + model.agv_path_cost(agv_steps_shelf_to_goal, payload_kg=model.pkg_weight_mu)
            + model.agv_load_cost
            + model.agv_path_cost(agv_steps_to_charger, payload_kg=0.0)
        )
        picker_cycle_cost = (
            model.picker_path_cost(picker_steps_to_shelf, n_lifts=1)
            + model.picker_path_cost(picker_steps_to_charger, n_lifts=0)
        )

        agv_needs_charge = agv_battery < max(self.low_battery_threshold, agv_cycle_cost)
        picker_needs_charge = picker_battery < max(self.low_battery_threshold, picker_cycle_cost)

        if self.strategy == ChargingStrategy.SYNCHRONIZED:
            if agv_needs_charge or picker_needs_charge:
                self.metrics.joint_charge_decisions += 1
                return {"agv": True, "picker": True}
            return {"agv": False, "picker": False}

        self.metrics.agv_charge_decisions += int(agv_needs_charge)
        self.metrics.picker_charge_decisions += int(picker_needs_charge)
        return {"agv": agv_needs_charge, "picker": picker_needs_charge}
