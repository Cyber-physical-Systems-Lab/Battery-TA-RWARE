from enum import Enum, IntEnum


class AgentType(Enum):
    AGV = 0
    PICKER = 1


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


class CollisionLayers(IntEnum):
    AGVS = 0
    PICKERS = 1
    SHELVES = 2
    CARRIED_SHELVES = 3


class PackageType(IntEnum):
    SOLO = 0       # 1 AGV, 0 pickers  -> AGV can complete alone
    STANDARD = 1   # 1 AGV, 1 picker   -> current default, lock-step pair
    LARGE = 2      # 1 AGV, 2 pickers -> heavy/large: one AGV carries it (a
                   # costly single-AGV haul), but two pickers must rendezvous to
                   # help load it at pickup AND to help place it back on return.


PACKAGE_REQUIREMENTS = {
    PackageType.SOLO:     (1, 0),
    PackageType.STANDARD: (1, 1),
    PackageType.LARGE:    (1, 2),
}