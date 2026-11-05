"""Centralized MRTA planner baselines plugged into the heuristic executor.

Both planners implement the ``assignment_fn`` interface of
``tarware.heuristic.heuristic_episode`` (and ``on_event`` for the
``assignment_listener`` hook), so they share the rule baseline's mission
execution, charging override, logging, and metrics exactly.
"""

from .bmrta import BMRTAPlanner
from .calvo import CalvoPlanner
from .cbba import CBBAPlanner
from .task_model import BasePlanner

PLANNER_NAMES = ("bmrta", "calvo", "cbba")


def make_planner(name: str, **kwargs) -> BasePlanner:
    if name == "bmrta":
        return BMRTAPlanner(**kwargs)
    if name == "calvo":
        return CalvoPlanner(**kwargs)
    if name == "cbba":
        return CBBAPlanner(**kwargs)
    raise ValueError(f"unknown planner '{name}' (expected one of {PLANNER_NAMES})")


__all__ = [
    "BasePlanner",
    "BMRTAPlanner",
    "CalvoPlanner",
    "CBBAPlanner",
    "PLANNER_NAMES",
    "make_planner",
]
