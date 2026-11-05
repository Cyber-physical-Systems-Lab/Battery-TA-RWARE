"""
Shared enums and package constants for the Tarware environment.

The definitions cover agent roles, primitive actions, directions, reward
aggregation modes, grid collision layers, package flow direction, and the
SOLO/STANDARD/HEAVY/PICKER_SOLO/LARGE package requirements and reward scaling
used by the warehouse, heuristic controller, action masks, and experiments.
"""

from enum import Enum, IntEnum


class AgentType(Enum):
    AGV = 0
    PICKER = 1
    AGENT = 2

class Action(Enum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    TOGGLE_LOAD = 4
    CHARGE = 5

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2

class CollisionLayers(IntEnum):
    AGVS = 0
    PICKERS = 1
    SHELVES = 2
    CARRIED_SHELVES = 3


class PackageDirection(Enum):
    OUT = 0   # shelf slot → goal dock (outbound delivery)
    IN  = 1   # goal dock → shelf slot (inbound restock)


class PackageType(IntEnum):
    SOLO = 0         # 1 AGV, 0 pickers  -> AGV can complete alone
    STANDARD = 1     # 1 AGV, 1 picker   -> lock-step cross-type pair
    LARGE = 2        # 2 AGVs, 2 pickers -> requires coordinated team
    HEAVY = 3        # 1 AGV, 2 pickers  -> heavy but single-AGV transport
    PICKER_SOLO = 4  # 0 AGVs, 1 picker  -> picker retrieves and carries alone


PACKAGE_REQUIREMENTS = {
    PackageType.SOLO:        (1, 0),
    PackageType.STANDARD:    (1, 1),
    PackageType.LARGE:       (2, 2),
    PackageType.HEAVY:       (1, 2),
    PackageType.PICKER_SOLO: (0, 1),
}

PACKAGE_REWARD_SCALE = {
    PackageType.SOLO:        0.5,
    PackageType.STANDARD:    1.0,
    PackageType.LARGE:       2.0,
    PackageType.HEAVY:       1.5,
    PackageType.PICKER_SOLO: 0.5,
}
