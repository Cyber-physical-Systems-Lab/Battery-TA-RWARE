"""Read-only state projection (contracts v1, section 3).

A StateView is a value: built on demand from a Warehouse, holding plain ints / floats / tuples and no
reference to live simulator objects. Building one draws no random numbers and mutates nothing, so it
may be called at any point without changing a trajectory (checked against G0 fingerprints).
It is not a restorable snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

SCHEMA_VERSION = 1

Cell = Tuple[int, int]


def _v(x) -> int:
    return int(getattr(x, "value", x))


@dataclass(frozen=True)
class RobotView:
    id: int
    kind: int                 # AgentType.value: 0 AGV, 1 PICKER
    y: int
    x: int
    dir: int                  # Direction.value: UP 0, DOWN 1, LEFT 2, RIGHT 3
    battery: float            # state of charge, percent [0, 100]
    busy: bool
    charging: bool            # charge intent flag (next idle step issues CHARGE)
    carrying_shelf: int       # shelf id, 0 = none
    target: int               # macro action id currently pursued, 0 = none
    path: Tuple[Cell, ...]    # remaining cells, stored as (x, y) -- kernel convention, not swapped


@dataclass(frozen=True)
class ShelfView:
    id: int
    y: int
    x: int
    package_type: int         # PackageType.value: SOLO 0, STANDARD 1, LARGE 2
    weight_kg: float
    required_agvs: int
    required_pickers: int


@dataclass(frozen=True)
class LayoutView:
    """Static per episode (fixed at construction): build once, reuse every tick."""
    schema_version: int
    grid_shape: Tuple[int, int]             # (H, W)
    highways: bytes                         # uint8 0/1, row-major [y][x], len H*W
    goals: Tuple[Cell, ...]                 # (x, y) as stored by the kernel
    chargers: Tuple[Cell, ...]              # (y, x)
    action_coords: Tuple[Tuple[int, int, int], ...]   # (action_id, y, x), sorted by action_id


@dataclass(frozen=True)
class StateView:
    schema_version: int
    tick: int
    grid_shape: Tuple[int, int]             # (H, W)
    occupancy: Tuple[bytes, ...]            # one per CollisionLayers (AGVS, PICKERS, SHELVES,
                                            # CARRIED_SHELVES): little-endian int32 entity ids,
                                            # row-major [y][x], 0 = empty
    robots: Tuple[RobotView, ...]
    shelves: Tuple[ShelfView, ...]
    request_queue: Tuple[int, ...]          # shelf ids, queue order
    goals: Tuple[Cell, ...]                 # (x, y) as stored by the kernel
    chargers: Tuple[Cell, ...]              # (y, x)

    def layer(self, index: int) -> np.ndarray:
        """Read-only (H, W) int32 view of one occupancy layer."""
        a = np.frombuffer(self.occupancy[index], dtype="<i4").reshape(self.grid_shape)
        a.flags.writeable = False
        return a

    def robot(self, robot_id: int) -> RobotView:
        for r in self.robots:
            if r.id == robot_id:
                return r
        raise KeyError(robot_id)


def build_layout_view(env) -> LayoutView:
    env = getattr(env, "unwrapped", env)
    H, W = (int(d) for d in env.grid_size)
    return LayoutView(
        schema_version=SCHEMA_VERSION, grid_shape=(H, W),
        highways=np.ascontiguousarray(env.highways != 0, dtype=np.uint8).tobytes(),
        goals=tuple((_v(g[0]), _v(g[1])) for g in env.goals),
        chargers=tuple((_v(c.y), _v(c.x)) for c in env.charging_stations),
        action_coords=tuple(sorted((int(k), _v(v[0]), _v(v[1])) for k, v in env.action_id_to_coords_map.items())))


def build_state_view(env) -> StateView:
    env = getattr(env, "unwrapped", env)
    H, W = (int(d) for d in env.grid_size)
    occupancy = tuple(np.ascontiguousarray(env.grid[i], dtype="<i4").tobytes() for i in range(env.grid.shape[0]))
    robots = tuple(
        RobotView(id=_v(a.id), kind=_v(a.type), y=_v(a.y), x=_v(a.x), dir=_v(a.dir),
                  battery=float(a.battery), busy=bool(a.busy), charging=bool(a.charging),
                  carrying_shelf=_v(a.carrying_shelf.id) if a.carrying_shelf is not None else 0,
                  target=_v(a.target),
                  path=tuple((_v(p[0]), _v(p[1])) for p in (a.path or ())))
        for a in env.agents)
    shelves = tuple(
        ShelfView(id=_v(s.id), y=_v(s.y), x=_v(s.x), package_type=_v(s.package_type),
                  weight_kg=float(s.weight), required_agvs=_v(s.required_agvs),
                  required_pickers=_v(s.required_pickers))
        for s in env.shelfs)
    return StateView(
        schema_version=SCHEMA_VERSION, tick=int(env._cur_steps), grid_shape=(H, W), occupancy=occupancy,
        robots=robots, shelves=shelves,
        request_queue=tuple(_v(s.id) for s in env.request_queue),
        goals=tuple((_v(g[0]), _v(g[1])) for g in env.goals),
        chargers=tuple((_v(c.y), _v(c.x)) for c in env.charging_stations))
