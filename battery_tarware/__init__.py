"""Battery-TA-RWARE: heterogeneous, battery-constrained multi-robot warehouse simulator.

    import battery_tarware
    env = battery_tarware.make("intermediate")              # or a dict / ScenarioConfig / .yaml path
    obs, info = env.reset(seed=0)

    import gymnasium as gym
    env = gym.make("battery_tarware/Intermediate-v2")       # registered on import
"""
from __future__ import annotations

__version__ = "2.1.0"

from gymnasium.envs.registration import register, registry

from battery_tarware.config import (
    ChargerTypeSpec, ConfigError, Distribution, DemandStream, EnergySpec, FailureSpec, JobSpec, JobTaskSpec,
    MapSpec, ObservationSpec, RewardSpec, RobotGroupSpec, RobotTypeSpec, RoleSpec, ScenarioConfig, StageSpec,
    TaskTypeSpec,
)
from battery_tarware.coalition import Coalition, feasible_coalitions
from battery_tarware.coordination import RULES, GreedyCoalitionCoordinator, RandomCoordinator, run_episode
from battery_tarware.env import BatteryTarwareEnv
from battery_tarware.layout import Layout, generate_ascii
from battery_tarware.scenarios import SCENARIOS, get_scenario, resolve_scenario
from battery_tarware.simulator import PRIMITIVE_ACTIONS, Simulator
from battery_tarware.state_view import StateView, build_state_view
from battery_tarware.tracking import PackageTracker, package_table, robot_table, status_counts, write_csv


def make(scenario="basic", render_mode=None, **overrides) -> BatteryTarwareEnv:
    """Build an environment from a built-in scenario name, a dict, a ScenarioConfig or a .json/.yaml
    path; keyword overrides replace top-level scenario fields (e.g. ``action_mode="primitive"``)."""
    return BatteryTarwareEnv(scenario, render_mode=render_mode, **overrides)


ENV_IDS = {"basic": "battery_tarware/Basic-v2", "intermediate": "battery_tarware/Intermediate-v2",
           "advanced": "battery_tarware/Advanced-v2", None: "battery_tarware/Warehouse-v2"}

for _name, _id in ENV_IDS.items():
    if _id not in registry:
        register(id=_id, entry_point="battery_tarware.env:BatteryTarwareEnv",
                 kwargs={} if _name is None else {"scenario": _name},
                 max_episode_steps=None, disable_env_checker=False)

__all__ = [
    "make", "BatteryTarwareEnv", "Simulator", "ScenarioConfig", "get_scenario", "resolve_scenario", "SCENARIOS",
    "ENV_IDS", "RobotTypeSpec", "RobotGroupSpec", "EnergySpec", "FailureSpec", "ChargerTypeSpec", "TaskTypeSpec",
    "RoleSpec", "StageSpec", "Distribution", "JobSpec", "JobTaskSpec", "DemandStream", "MapSpec", "RewardSpec",
    "ObservationSpec", "ConfigError", "Coalition", "feasible_coalitions", "RULES", "GreedyCoalitionCoordinator",
    "RandomCoordinator", "run_episode", "StateView", "build_state_view", "Layout", "generate_ascii",
    "PRIMITIVE_ACTIONS", "PackageTracker", "package_table", "robot_table", "status_counts", "write_csv", "__version__",
]
