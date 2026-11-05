import random
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx
import numpy as np
from battery_tarware.astar import astar_path
from battery_tarware.definitions import (Action, AgentType, Direction,
                                 CollisionLayers, PackageType,
                                 PACKAGE_REQUIREMENTS)
from battery_tarware.utils import find_sections, get_next_micro_action

_FIXING_CLASH_TIME = 4
_STUCK_THRESHOLD = 5
_BATTERY_FULL = 100

# Movement/charge energy, calibrated to an OMRON LD-250-class spec (25.6V, 72Ah -> 1.843kWh):
# empty(unloaded)-drive ~142W -> 13.0h runtime; full-load-drive ~184W -> 10.0h runtime;
# fast charger 20%->80% in 130min. One sim step = 5.0 real seconds (_SIM_STEP_SECONDS in
# planner/coordinator.py -- promoted from a cosmetic reporting constant to the actual basis
# for these rates), so every rate below is minutes-per-real-spec * (5/60) min/step, not a
# guess: a full loaded discharge is exactly 7200 steps (=10h), 20%->80% charge is exactly
# 1560 steps (=130min), and 24h of episode is exactly 17280 steps.
_STEP_MINUTES = 5.0 / 60.0
_BATTERY_CONSUMPTION_MOVE = (1.0 / 600.0) * 100.0 * _STEP_MINUTES         # AGV, carrying a shelf (~0.013889)
_BATTERY_CONSUMPTION_MOVE_UNLOADED = _BATTERY_CONSUMPTION_MOVE * (10.0 / 13.0)   # AGV, not carrying (~0.010684)
_BATTERY_CONSUMPTION_LOAD = 2.0 * _BATTERY_CONSUMPTION_MOVE               # one-off pick/place toggle cost; no
                                                                            # hardware spec covers this action, so
                                                                            # its RATIO to a move (2x) is preserved
                                                                            # from the pre-recalibration value while
                                                                            # its scale follows the new step size
_BATTERY_CHARGE_RATE = _BATTERY_CONSUMPTION_MOVE * (36.0 / 13.0)          # fast charger: 20%->80%/130min (~0.038462)
KINETIC_CONSUMPTION = _BATTERY_CONSUMPTION_MOVE
LOADED_FACTOR = _BATTERY_CONSUMPTION_MOVE / _BATTERY_CONSUMPTION_MOVE_UNLOADED   # per-move loaded:unloaded ratio (=1.3)
STANDBY_CONSUMPTION = 0.1

# Heterogeneous energy model parameters
_PKG_WEIGHT_MU = 5.0       # Mean package weight (kg)
_PKG_WEIGHT_SIGMA = 2.0    # Package weight std dev (kg)
_PKG_WEIGHT_MIN = 0.5      # Minimum package weight (kg)
_PKG_WEIGHT_MAX = 20.0     # Maximum package weight (kg)
_AGV_WEIGHT_FACTOR = 0.04  # Fractional extra energy per kg when loaded moving (dimensionless, scale-invariant)
_PICKER_LIFT_ENERGY = 1.0 * _BATTERY_CONSUMPTION_MOVE  # Extra picker energy per lift assist action (ratio preserved)

class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.x = x
        self.y = y

class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int, dir_: Direction, agent_type: AgentType, battery: Optional[float] = None):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)
        self.dir = dir_
        self.req_action: Optional[Action] = None
        self.carrying_shelf: Optional[Shelf] = None
        self.canceled_action = None
        self.has_delivered = False
        self.path: Optional[List[Tuple[int, int]]] = None
        self.busy = False
        self.fixing_clash = 0
        self.type = agent_type
        self.target = 0
        self.battery = float(random.randint(20, _BATTERY_FULL) if battery is None else battery)
        self.charging = False
        self.total_lifts = 0       # Cumulative lift assists (for picker energy tracking)

    def req_location(self, grid_size) -> Tuple[int, int]:
        if self.req_action != Action.FORWARD:
            return self.x, self.y
        elif self.dir == Direction.UP:
            return self.x, max(0, self.y - 1)
        elif self.dir == Direction.DOWN:
            return self.x, min(grid_size[0] - 1, self.y + 1)
        elif self.dir == Direction.LEFT:
            return max(0, self.x - 1), self.y
        elif self.dir == Direction.RIGHT:
            return min(grid_size[1] - 1, self.x + 1), self.y

        raise ValueError(
            f"Direction is {self.dir}. Should be one of {[v for v in Direction]}"
        )

    def req_direction(self) -> Direction:
        wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.req_action == Action.RIGHT:
            return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        elif self.req_action == Action.LEFT:
            return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        else:
            return self.dir

class Shelf(Entity):
    counter = 0

    def __init__(self, x, y, package_type: PackageType = PackageType.STANDARD):
        Shelf.counter += 1
        super().__init__(Shelf.counter, x, y)
        self.package_type = package_type
        self.required_agvs, self.required_pickers = PACKAGE_REQUIREMENTS[package_type]
        # Carrier group: AGVs that participated in the atomic load (lead + helpers).
        # Empty until a successful TOGGLE_LOAD; cleared on unload.
        self.carrier_group: List["Agent"] = []
        # Package weight sampled from truncated normal distribution, biased by size.
        weight_bias = {PackageType.SOLO: 0.5, PackageType.STANDARD: 1.0,
                       PackageType.LARGE: 1.75}[package_type]
        self.weight = float(np.clip(
            np.random.normal(_PKG_WEIGHT_MU * weight_bias, _PKG_WEIGHT_SIGMA),
            _PKG_WEIGHT_MIN, _PKG_WEIGHT_MAX
        ))

class ChargingStation(Entity):
    counter = 0

    def __init__(self, x, y):
        ChargingStation.counter += 1
        super().__init__(ChargingStation.counter, x, y)

class StuckCounter:
    def __init__(self, position: Tuple[int, int]):
        self.position = position
        self.count = 0

    def update(self, new_position: Tuple[int, int]):
        if new_position == self.position:
            self.count += 1
        else:
            self.count = 0
            self.position = new_position

    def reset(self, position=None):
        self.count = 0
        if position:
            self.position = position

class Warehouse:

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        shelf_columns: int,
        column_height: int,
        shelf_rows: int,
        num_agvs: int,
        num_pickers: int,
        request_queue_size: int,
        max_inactivity_steps: Optional[int],
        max_steps: Optional[int],
        package_distribution: Optional[Dict] = None,
        charger_rows: int = 1,
        replenish_requests: bool = False,
        bottom_charger_rows: int = 0,
        charger_spread: bool = False,
        charger_all_rows: bool = False,
        layout: str = "classic",
        rack_width: int = 1,
        lane_width: int = 2,
        initial_battery_profile: str = "legacy",
        initial_battery_low: float = 20.0,
        initial_battery_high: float = 100.0,
        battery_drain: bool = True,
        picker_energy_factor: float = 1.25,
    ):
        """The robotic warehouse environment

        Creates a grid world where multiple agents (robots)
        are supposed to collect shelfs, bring them to a goal
        and then return them.
        .. note:
            The grid looks like this:

            shelf
            columns
                vv
            ----------
            -XX-XX-XX-        ^
            -XX-XX-XX-  Column Height
            -XX-XX-XX-        v
            ----------
            -XX----XX-   <\
            -XX----XX-   <- Shelf Rows
            -XX----XX-   </
            ----------
            ----GG----

            G: is the goal positions where
            they bring the correct shelfs.

            The final grid size will be
            height: (column_height + 1) * shelf_rows + 2
            width: (2 + 1) * shelf_columns + 1

            The bottom-middle column will be removed to allow for
            robot queuing next to the goal locations

        :param shelf_columns: Number of columns in the warehouse
        :type shelf_columns: int
        :param column_height: Column height in the warehouse
        :type column_height: int
        :param shelf_rows: Number of columns in the warehouse
        :type shelf_rows: int
        :param num_agvs: Number of spawned and controlled agv
        :type num_agvs: int
        :param num_pickers: Number of spawned and controlled pickers
        :type num_pickers: int
        :param request_queue_size: How many shelfs are simultaneously requested
        :type request_queue_size: int
        :param max_inactivity: Number of steps without a delivered shelf until environment finishes
        :type max_inactivity: Optional[int]
        :param max_steps: Maximum number of steps in the episode
        :type max_steps: Optional[int]
        :param picker_energy_factor: Picker per-move battery cost as a multiple of the AGV-loaded
            reference rate. OMRON-spec-derived range is 1.1-1.4x; 1.25 is the midpoint default.
        :type picker_energy_factor: float
        """

        self.goals: List[Tuple[int, int]] = []

        self.num_agvs = num_agvs
        self.num_pickers = num_pickers
        self.num_agents = num_agvs + num_pickers
        self.charger_rows = max(1, int(charger_rows))
        # Opt-in: also place chargers in the bottom highway rows (near the goal
        # row), not just the top. On tall maps the full pick->deliver->return->
        # recharge cycle exceeds battery because every charger sits at the top,
        # ~map-height away from the bottom goals, so AGVs strand mid-cycle and
        # throughput collapses. Bottom chargers let an AGV top up near where it
        # delivers, restoring reachability on extralarge/huge. Default 0 keeps
        # the original top-only layout byte-for-byte.
        self.bottom_charger_rows = max(0, int(bottom_charger_rows))
        # When bottom_charger_rows > 0, the original packing was column-major
        # (fill every charger row of column x, then x+1), which clusters all
        # chargers in the leftmost ceil(num_agents / n_charger_rows) columns and
        # leaves the right side of the map with no charger near its goals. With
        # charger_spread=True the SAME number of chargers is instead placed in
        # evenly-spaced columns across the full width, so every bottom goal has a
        # charger within a few cells — cutting charge-trip travel and mid-cycle
        # strandings on the deep extralarge/huge maps. Default False preserves
        # the original column-major layout (and top-only layouts are unaffected).
        self.charger_spread = bool(charger_spread)
        # When True, distribute the (unchanged) num_agents charger budget across
        # ALL usable highway bands — top, MIDDLE, and bottom — instead of only the
        # top charger_rows + bottom bottom_charger_rows. On deep maps the middle
        # bands (where the shelves are) are otherwise charger-less, so an AGV that
        # drains while picking mid-map must travel far to a charger (~64% of its
        # charging-mission time is travel). Mid-map chargers put a station within a
        # few cells of any pick/return point, cutting charge-travel. Implies the
        # full-width spread packing. Charger COUNT and goals are unchanged. Default
        # False; top-only (bottom_charger_rows=0, no all_rows) stays byte-for-byte.
        self.charger_all_rows = bool(charger_all_rows)

        if layout not in ("classic", "tiled"):
            raise ValueError(f"Unknown layout '{layout}' (expected 'classic' or 'tiled')")
        self.layout = layout
        # One-way lanes (see _make_layout_tiled); classic keeps bidirectional 2-wide highways.
        self._move_ok: Optional[np.ndarray] = None
        if layout == "tiled":
            self._make_layout_tiled(shelf_columns, shelf_rows, column_height, rack_width, lane_width)
        else:
            self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
        self._charging_station_positions: set = {(s.x, s.y) for s in self.charging_stations}
        self.shelf_action_ids = list(range(len(self.goals) + 1, len(self.action_id_to_coords_map) - len(self.charging_stations) + 1))
        # If no Pickers are generated, AGVs can perform picks independently
        if num_pickers > 0:
            self._agent_types = [AgentType.AGV for _ in range(num_agvs)] + [AgentType.PICKER for _ in range(num_pickers)]
        else:
            self._agent_types = [AgentType.AGV for _ in range(self.num_agents)]

        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        # Initialize inactive steps counter to 0 to avoid comparisons with None before reset()
        self._cur_inactive_steps = 0
        self._cur_steps = 0
        self.max_steps = max_steps

        self.request_queue_size = request_queue_size
        # When True the queue is kept at a constant level: each delivered shelf
        # is replaced by a freshly-sampled request (continuous-arrival /
        # Poisson-source warehouse). Default False preserves the original
        # "clear a fixed batch" semantics and the episode-ends-on-empty-queue
        # behaviour used by existing baselines + C++/Python parity.
        self.replenish_requests = bool(replenish_requests)
        self.request_queue = []
        self.rack_groups = find_sections(list([loc for loc in self.action_id_to_coords_map.values() if (loc[1], loc[0]) not in self.goals]))
        self.agents: List[Agent] = []
        self.stuck_counters = []

        self._package_distribution = self._normalise_package_distribution(package_distribution)
        self.initial_battery_profile = str(initial_battery_profile or "legacy")
        self.initial_battery_low = float(initial_battery_low)
        self.initial_battery_high = float(initial_battery_high)
        # When False, movement/load/lift never consume SoC (charging still
        # works but is never needed). Combined with
        # initial_battery_profile="constant", initial_battery_high=100 this is
        # the infinite-energy ORACLE upper bound: the same allocator runs with
        # the energy problem deleted, bounding what perfect energy management
        # could ever recover. Default True = physical battery model.
        self.battery_drain = bool(battery_drain)
        self.picker_energy_factor = float(picker_energy_factor)
        self.renderer = None

    @property
    def targets_agvs(self):
        return [agent.target for agent in self.agents[:self.num_agvs]]

    @property
    def targets_pickers(self):
        return [agent.target for agent in self.agents[self.num_agvs:]]

    def _make_layout_from_params(self, shelf_columns: int, shelf_rows: int, column_height: int) -> None:
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"
        self._bottom_rows = 2
        self._highway_lanes = 2
        self.column_width = 2
        self.column_height = column_height
        self.grid_size = (
            self._highway_lanes + (self.column_height + self._highway_lanes) * shelf_rows  + self._bottom_rows + 1,
            self._highway_lanes + (self.column_width  + self._highway_lanes) * shelf_columns,
        )
        self.grid = np.zeros((len(CollisionLayers), *self.grid_size), dtype=np.int32)

        def get_highway_lanes_indices(axis_size, step):
            return [
                i + j
                for i in range(
                    0, axis_size, step + self._highway_lanes
                )
                for j in range(self._highway_lanes)
            ]

        highway_ys = get_highway_lanes_indices(self.grid_size[0], self.column_height)
        highway_xs = get_highway_lanes_indices(self.grid_size[1], self.column_width)

        def highway_func(x, y):
            return x in highway_xs or y in highway_ys or y >= self.grid_size[0] - 1 - self._bottom_rows

        self.goals = [
            (i, self.grid_size[0] - 1)
            for i in range(self.grid_size[1]) if not i in highway_xs
        ]
        self.num_goals = len(self.goals)

        # Charging stations occupy highway cells. We walk the full list of
        # highway rows (there is a 2-lane band between every pair of shelf
        # rows) top-to-bottom and pack chargers along each until the agent
        # count is satisfied. ``charger_rows`` caps how many distinct rows we
        # consume — default 1 reproduces the original single-row layout.
        # Bottom row (row containing goals) is excluded so chargers don't
        # collide with delivery points.
        usable_rows = [
            y for y in highway_ys
            if y < self.grid_size[0] - 1 - self._bottom_rows  # skip band next to goals
        ]
        if self.bottom_charger_rows > 0 or self.charger_all_rows:
            # Distribute chargers across top AND bottom highway rows. Pack
            # column-major (x outer, row inner) so both bands get chargers even
            # when num_agents is small — guarantees goal-reachable chargers.
            if self.charger_all_rows:
                # Use EVERY usable highway band (top + middle + bottom) so the
                # whole vertical extent has chargers, not just the ends.
                rows_to_use = list(usable_rows)
            else:
                top = usable_rows[: self.charger_rows]
                bottom = usable_rows[-self.bottom_charger_rows:]
                rows_to_use = sorted(set(top) | set(bottom))
            width = self.grid_size[1]
            # charger_all_rows implies the full-width spread (otherwise column-
            # major would cluster all bands into the leftmost columns again).
            if self.charger_spread or self.charger_all_rows:
                # Same column-major packing, but the columns we consume are
                # spread evenly across the FULL width instead of clustered on
                # the left. We need ceil(num_agents / n_rows) columns; place
                # them at evenly-spaced x so each charger row (in particular the
                # bottom rows next to the goals) covers the whole width. Falls
                # back to the remaining columns if rounding collapses any.
                n_rows = max(1, len(rows_to_use))
                n_cols = -(-self.num_agents // n_rows)  # ceil
                if n_cols >= width:
                    x_order = list(range(width))
                else:
                    spread = sorted({(2 * j + 1) * width // (2 * n_cols)
                                     for j in range(n_cols)})
                    used = set(spread)
                    for x in range(width):  # pad if rounding produced dupes
                        if len(spread) >= n_cols:
                            break
                        if x not in used:
                            spread.append(x)
                            used.add(x)
                    x_order = sorted(spread)
                    for x in range(width):  # ensure every column is reachable
                        if x not in used:    # as a last-resort fallback
                            x_order.append(x)
            else:
                x_order = range(width)
            station_cells = []
            for x in x_order:
                for y in rows_to_use:
                    station_cells.append((x, y))
                    if len(station_cells) >= self.num_agents:
                        break
                if len(station_cells) >= self.num_agents:
                    break
        else:
            rows_to_use = usable_rows[: self.charger_rows]
            station_cells = []
            for y in rows_to_use:
                for x in range(self.grid_size[1]):
                    station_cells.append((x, y))
                    if len(station_cells) >= self.num_agents:
                        break
                if len(station_cells) >= self.num_agents:
                    break
        if len(station_cells) < self.num_agents:
            raise ValueError(
                f"Warehouse layout cannot fit {self.num_agents} chargers: "
                f"only {len(station_cells)} slots available across "
                f"{len(rows_to_use)} highway row(s). Increase shelf_columns, "
                f"shelf_rows, or charger_rows."
            )
        self.charging_stations = [ChargingStation(x, y) for (x, y) in station_cells[:self.num_agents]]

        self.highways = np.zeros(self.grid_size, dtype=np.int32)
        self.action_id_to_coords_map = {i+1: (x, y) for i, (y, x) in enumerate(self.goals)}
        item_loc_index=len(self.action_id_to_coords_map)+1
        for x in range(self.grid_size[1]):
            for y in range(self.grid_size[0]):
                self.highways[y, x] = highway_func(x, y)
                if not highway_func(x, y) and (x, y) not in self.goals:
                    self.action_id_to_coords_map[item_loc_index] = (y, x)
                    item_loc_index+=1
        for charging_station in self.charging_stations:
            self.action_id_to_coords_map[item_loc_index] = (charging_station.y, charging_station.x)
            item_loc_index += 1

    def _make_layout_tiled(self, tile_cols: int, tile_rows: int, column_height: int,
                           rack_width: int = 1, lane_width: int = 2) -> None:
        """Tiled, traffic-stable layout (opt-in; the classic layout is untouched).

        A tile is one rack block (``rack_width`` wide x ``column_height`` high) with its lanes. Per block:
          * the top ``dock_rows`` rows are candidate DOCK cells (chargers that double as parking bays,
            off the lanes); exactly ``num_agents`` docks are used, spread round-robin over blocks; unused
            candidates become ordinary shelf cells;
          * the bottom-left cell is a GOAL, so goal/service capacity scales with the tile count and every
            delivery is served near where the shelf was picked.
        With ``rack_width=1`` every shelf has TWO adjacent highway cells (one per side), so a LARGE task can
        place its two pickers at distinct 4-adjacent cells (the planner's rendezvous rule).
        Highways are ``2*lane_width`` cells wide, ``lane_width`` lanes per direction: horizontal bands run
        West (upper rows) / East (lower rows), vertical highways South (left columns) / North (right
        columns). Moves between highway cells must follow the lane direction, plus lateral moves between
        lanes of the SAME direction (overtaking a stopped robot in the curb lane). Entering/leaving rack,
        dock and goal cells from a lane is unrestricted.
        """
        h, w, lw = int(column_height), int(rack_width), int(lane_width)
        # Dock (charger/parking-bay) candidates per block: enough rows to hold one dock per agent,
        # never fewer than 5 and never as low as the block's goal row. Only the first ``num_agents``
        # candidates (slot-major, so row 0 of every block first) become real docks; the rest stay
        # shelves. Growing this therefore leaves smaller fleets' maps identical.
        n_blocks = max(1, int(tile_rows) * int(tile_cols))
        dock_rows = min(h - 1, max(5, -(-self.num_agents // (n_blocks * w))))
        hwd = 2 * lw                       # highway width in cells
        self._bottom_rows = 0
        self._highway_lanes = hwd
        self.column_width = w
        self.column_height = h
        band, pitch = h + hwd, hwd + w
        H, W = hwd + band * tile_rows, hwd + pitch * tile_cols
        self.grid_size = (H, W)
        self.grid = np.zeros((len(CollisionLayers), *self.grid_size), dtype=np.int32)
        hw_ys = {b * band + j for b in range(tile_rows + 1) for j in range(hwd)}
        hw_xs = {k * pitch + j for k in range(tile_cols + 1) for j in range(hwd)}

        goals, dock_pool = [], []
        for r in range(tile_rows):
            y_top = hwd + r * band
            y_bot = y_top + h - 1
            for k in range(tile_cols):
                x_l = hwd + pitch * k
                goals.append((x_l, y_bot))
                dock_pool.append([(x_l + dx, y_top + j) for j in range(dock_rows) for dx in range(w)])
        self.goals = goals
        self.num_goals = len(goals)
        n_slots = len(dock_pool[0])
        docks = []
        for j in range(n_slots):
            for block in dock_pool:
                if len(docks) < self.num_agents:
                    docks.append(block[j])
        if len(docks) < self.num_agents:
            raise ValueError(
                f"Tiled layout {tile_rows}x{tile_cols} has {len(dock_pool) * n_slots} dock slots < {self.num_agents} agents"
            )
        self.charging_stations = [ChargingStation(x, y) for (x, y) in docks]

        self.highways = np.zeros(self.grid_size, dtype=np.int32)
        for y in range(H):
            for x in range(W):
                self.highways[y, x] = int(x in hw_xs or y in hw_ys)

        goal_set, dock_set = set(goals), set(docks)
        self.action_id_to_coords_map = {i + 1: (gy, gx) for i, (gx, gy) in enumerate(goals)}
        item_loc_index = len(goals) + 1
        for x in range(W):
            for y in range(H):
                if not self.highways[y, x] and (x, y) not in goal_set and (x, y) not in dock_set:
                    self.action_id_to_coords_map[item_loc_index] = (y, x)
                    item_loc_index += 1
        for cs in self.charging_stations:
            self.action_id_to_coords_map[item_loc_index] = (cs.y, cs.x)
            item_loc_index += 1

        # One-way lanes: move_ok[d, y, x], d in (+y, -y, +x, -x) matching astar's neighbour order.
        # Column lane group of x: (x % pitch) // lw -> 0: South, 1: North; row group of y: (y % band) // lw
        # -> 0: West, 1: East.
        def col_group(x):
            return (x % pitch) // lw if x in hw_xs else None

        def row_group(y):
            return (y % band) // lw if y in hw_ys else None

        ok = np.ones((4, H, W), dtype=bool)
        for y in range(H):
            for x in range(W):
                if not self.highways[y, x]:
                    continue
                cg, rg = col_group(x), row_group(y)
                for d, (dy, dx) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < H and 0 <= nx < W and self.highways[ny, nx]):
                        continue
                    allowed = False
                    if cg is not None:
                        if d == (0 if cg == 0 else 1):                       # along the vertical lane
                            allowed = True
                        elif d in (2, 3) and col_group(nx) == cg:            # lateral, same-direction lane
                            allowed = True
                    if rg is not None:
                        if d == (3 if rg == 0 else 2):                       # along the horizontal lane
                            allowed = True
                        elif d in (0, 1) and row_group(ny) == rg:            # lateral, same-direction lane
                            allowed = True
                    ok[d, y, x] = allowed
        self._move_ok = ok

    def _is_highway(self, x: int, y: int) -> bool:
        return self.highways[y, x]

    @staticmethod
    def _normalise_package_distribution(dist) -> Dict[PackageType, float]:
        """Normalise a package-type distribution dict. Falls back to all STANDARD."""
        if not dist:
            return {PackageType.STANDARD: 1.0}
        normalised: Dict[PackageType, float] = {}
        for key, weight in dist.items():
            if weight <= 0:
                continue
            if isinstance(key, PackageType):
                pkg = key
            elif isinstance(key, int):
                pkg = PackageType(key)
            else:
                pkg = PackageType[str(key).upper()]
            normalised[pkg] = normalised.get(pkg, 0.0) + float(weight)
        if not normalised:
            return {PackageType.STANDARD: 1.0}
        total = sum(normalised.values())
        return {k: v / total for k, v in normalised.items()}

    def _sample_package_type(self) -> PackageType:
        types = list(self._package_distribution.keys())
        probs = list(self._package_distribution.values())
        return types[int(np.random.choice(len(types), p=probs))]

    def _initial_battery_for_agent(self, index: int, agent_type: AgentType) -> Optional[float]:
        """Optional benchmark-controlled initial SoC.

        ``legacy`` returns None so Agent keeps the historical random randint(20,100)
        behaviour and random-number stream. Other profiles are for H-calibrated
        benchmark construction and deliberately set the initial fleet SoC.
        """
        profile = self.initial_battery_profile.lower()
        lo = max(0.0, min(100.0, self.initial_battery_low))
        hi = max(0.0, min(100.0, self.initial_battery_high))
        if hi < lo:
            lo, hi = hi, lo
        if profile in ("", "legacy"):
            return None
        if profile in ("constant", "full"):
            return hi
        if profile == "uniform":
            return float(random.uniform(lo, hi))
        if profile == "bimodal":
            return lo if index % 2 == 0 else hi
        if profile == "agv_low":
            return lo if agent_type == AgentType.AGV else hi
        if profile == "picker_low":
            return lo if agent_type == AgentType.PICKER else hi
        raise ValueError(
            "Unknown initial_battery_profile "
            f"'{self.initial_battery_profile}'. Expected legacy, constant, "
            "uniform, bimodal, agv_low, or picker_low."
        )

    def find_path(self, start, goal: Tuple[int, int], agent, care_for_agents: bool = True, directed: bool = True) -> List[Tuple[int, int]]:
        """
        Constructs a path from start to goal using the A* algorithm contidioned on a grid that integrates both the
        environment specific (shelf) and the presence of other agents as obstacles.

        If `care_for_agents` is True, the grid is adjusted to consider other agents as obtacles. However, we avoid
        situatiosns where the paths is invalidated by agents of other types waiting to cooperate with the current.
        For Pickers, the grid is further modified to ensure they can only travel through designated highways and
        access goal locations.

        Parameters:
        - care_for_agents (bool): Whether to consider other agents in the grid.
        - directed (bool): Respect one-way lanes (tiled layout only). Planner cost ESTIMATES pass
          False so every allocator (and the C++ mirror) sees the same undirected distances; motion
          always uses the default True.
        - agent (Agent): The agent for which the path is being calculated.
        - start (tuple): The starting coordinates (x, y) of the agent.
        - goal (tuple): The goal coordinates (x, y) for the agent.

        Returns:
        - List of tuples representing the path from start to goal, or an empty list if no path is found.
        """
        grid = np.zeros(self.grid_size)
        grid += self.grid[CollisionLayers.SHELVES]
        if care_for_agents:
            grid += self.grid[CollisionLayers.AGVS]
            grid += self.grid[CollisionLayers.PICKERS]
        # Agents should start a path regardless if some others are waiting around the target location
        grid[goal[0], goal[1]] = 0

        if agent.type == AgentType.PICKER:
            # Pickers can only travel through the highway, but can access goal locations
            grid += (1-self.highways)
            grid[goal[0], goal[1]] -= not self._is_highway(goal[1], goal[0])
            if self.layout == "classic":
                for i in range(self.grid_size[1]):
                    grid[self.grid_size[0] - 1, i] = 1

        # Carrying AGVs must also stay on highways (they may still exit via the goal cell
        # or dock at a shelf/charging target, so exempt the goal cell itself).
        if agent.type == AgentType.AGV and agent.carrying_shelf is not None:
            grid += (1 - self.highways)
            grid[goal[0], goal[1]] -= not self._is_highway(goal[1], goal[0])

        # Ban Pickers crossing through racks if adjacent target location is chosen and force them thake the long way around.
        start_fix = (0, 0)
        if agent.type == AgentType.PICKER and  ((not self._is_highway(start[1], start[0])) and goal[0] == start[0] and abs(goal[1] - start[1]) == 1):
            if self._is_highway(start[1] - 1, start[0]):
                start_fix = (0, - 1)
            if self._is_highway(start[1] + 1, start[0]):
                start_fix = (0, 1)
            grid[start[0], start[1]] = 1

        grid[start[0]+start_fix[0], start[1]+start_fix[1]] = 0
        grid = [list(map(int, l)) for l in (grid!=0)]
        grid = np.array(grid, dtype=np.float32)
        grid[np.where(grid == 1)] = np.inf
        grid[np.where(grid == 0)] = 1
        adjusted_start = (start[0] + start_fix[0], start[1] + start_fix[1])
        result = astar_path(grid, adjusted_start, goal, allow_diagonal=False,
                            move_ok=self._move_ok if directed else None) # returns None if cant find path
        if result is not None:
            path_coords = [tuple(x) for x in list(result)]
            path_coords = path_coords[1 - int(grid[start[0], start[1]] > 1):]
        else:
            path_coords = []

        if path_coords:
            return [(x, y) for y, x in path_coords]
        else:
            return []

    def _recalc_grid(self) -> None:
        self.grid.fill(0)

        carried_shelf_ids = {agent.carrying_shelf.id for agent in self.agents if agent.carrying_shelf}
        for shelf in self.shelfs:
            if shelf.id not in carried_shelf_ids:
                self.grid[CollisionLayers.SHELVES, shelf.y, shelf.x] = shelf.id
        for agent in self.agents:
            layer = CollisionLayers.PICKERS if agent.type == AgentType.PICKER else CollisionLayers.AGVS
            self.grid[layer, agent.y, agent.x] = agent.id
            if agent.carrying_shelf:
                 self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = agent.carrying_shelf.id

    def get_carrying_shelf_information(self):
        return [agent.carrying_shelf != None for agent in self.agents[:self.num_agvs]]

    def get_shelf_request_information(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        request_item_map = np.zeros(len(self.shelfs), dtype=int)
        requested_shelf_ids = [shelf.id for shelf in self.request_queue]
        for id_ in self.shelf_action_ids:
            coords = self.action_id_to_coords_map[id_]
            if self.grid[CollisionLayers.SHELVES, coords[0], coords[1]] in requested_shelf_ids:
                request_item_map[id_ - len(self.goals) - 1] = 1
        return request_item_map

    def get_empty_shelf_information(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        empty_item_map = np.zeros(len(self.shelfs), dtype=int)
        for id_ in self.shelf_action_ids:
            coords = self.action_id_to_coords_map[id_]
            if self.grid[CollisionLayers.SHELVES, coords[0], coords[1]] == 0 and (
                self.grid[CollisionLayers.CARRIED_SHELVES, coords[0], coords[1]] == 0
                or self.agents[
                    self.grid[CollisionLayers.AGVS, coords[0], coords[1]] - 1
                ].req_action
                not in [Action.NOOP, Action.TOGGLE_LOAD]
            ):
                empty_item_map[id_ - len(self.goals) - 1] = 1
        return empty_item_map

    def attribute_macro_actions(self, macro_actions: List[int]) -> Tuple[int, int]:
        agvs_distance_travelled = 0
        pickers_distance_travelled = 0
        # Logic for Macro Actions
        for agent, macro_action in zip(self.agents, macro_actions):
            # Initialize action for step
            agent.req_action = Action.NOOP
            # Collision avoidance logic
            if agent.fixing_clash > 0:
                agent.fixing_clash -= 1
            if not agent.busy:
                agent.target = 0
                if macro_action != 0:
                    agent.path = self.find_path((agent.y, agent.x), self.action_id_to_coords_map[macro_action], agent, care_for_agents=False)
                    if agent.path:
                        agent.busy = True
                        agent.target = macro_action
                        agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                        self.stuck_counters[agent.id - 1].reset((agent.x, agent.y))
                    elif agent.type == AgentType.AGV:
                        # Already at target cell: keep issuing TOGGLE_LOAD each step
                        # until the load requirements are met (needed for LARGE tasks
                        # where helpers may arrive later).
                        agent.target = macro_action
                        agent.req_action = Action.TOGGLE_LOAD
            else:
                # Check if agent finished the given path, if not continue the path
                if agent.path is None or agent.path == []:
                    if agent.type == AgentType.AGV:
                        agent.req_action = Action.TOGGLE_LOAD
                    elif agent.type == AgentType.PICKER:
                        agent.busy = False
                else:
                    agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                    agvs_distance_travelled += int(agent.type == AgentType.AGV)
                    pickers_distance_travelled += int(agent.type == AgentType.PICKER)
                if agent.path is not None and len(agent.path) == 1:
                    # If agent is at the end of a path and carrying a shelf and the target location is already occupied, restart agent
                    if agent.carrying_shelf and self.grid[CollisionLayers.SHELVES, agent.path[-1][1], agent.path[-1][0]]:
                        agent.req_action = Action.NOOP
                        agent.busy = False
                    # Logic for Pickers to load shelves if AGV is present at location or wait otherwise
                    if agent.type == AgentType.PICKER:
                        last_x, last_y = agent.path[-1]
                        if (last_x, last_y) in self._charging_station_positions:
                            pass  # Charger target — no AGV handshake needed; let picker walk in
                        elif (
                            self.grid[CollisionLayers.AGVS, last_y, last_x] == 0
                            or self.agents[self.grid[CollisionLayers.AGVS, last_y, last_x] - 1].req_action != Action.TOGGLE_LOAD
                        ):
                            agent.req_action = Action.NOOP
                        elif (
                            self.grid[CollisionLayers.AGVS, last_y, last_x] != 0
                            and self.agents[self.grid[CollisionLayers.AGVS, last_y, last_x] - 1].req_action == Action.TOGGLE_LOAD
                        ):
                            self.stuck_counters[agent.id - 1].reset((agent.x, agent.y))
                            
            # Check if agent has finished its path and should charge
            if (agent.path is None or agent.path == []) and agent.charging and agent.battery < _BATTERY_FULL:
                agent.req_action = Action.CHARGE
                agent.charging = False
                agent.busy = False
        return agvs_distance_travelled, pickers_distance_travelled

    def resolve_move_conflict(self, agent_list):
        commited_agents = set()
        G = nx.DiGraph()
        for agent in agent_list:
            start = agent.x, agent.y
            target = agent.req_location(self.grid_size)
            if (target[0], target[1]) not in self.charging_stations:
                G.add_edge(start, target)
        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
        for comp in wcomps:
            try:
                # if we find a cycle in this component we have to
                # commit all nodes in that cycle, and nothing else
                cycle = nx.algorithms.find_cycle(comp)
                if len(cycle) == 2:
                    # we have a situation like this: [A] <-> [B]
                    # which is physically impossible. so skip
                    continue
                for edge in cycle:
                    start_node = edge[0]
                    agent_id = self.grid[CollisionLayers.AGVS, start_node[1], start_node[0]]
                    # action = self.agents[agent_id - 1].req_action
                    # print(f"{agent_id}: C {cycle} {action}")
                    if agent_id > 0:
                        commited_agents.add(agent_id)
                        continue
                    picker_id = self.grid[CollisionLayers.PICKERS, start_node[1], start_node[0]]
                    if picker_id > 0:
                        commited_agents.add(picker_id)
                        continue
            except nx.NetworkXNoCycle:
                longest_path = nx.algorithms.dag_longest_path(comp)
                for x, y in longest_path:
                    agent_id = self.grid[CollisionLayers.AGVS, y, x]
                    if agent_id:
                        commited_agents.add(agent_id)
                        continue
                    picker_id = self.grid[CollisionLayers.PICKERS, y, x]
                    if picker_id:
                        commited_agents.add(picker_id)
        clashes = 0
        for agent in agent_list:
            for other in agent_list:
                if agent.id != other.id:
                    agent_new_x, agent_new_y = agent.req_location(self.grid_size)
                    other_new_x, other_new_y = other.req_location(self.grid_size)
                    # Clash fixing logic
                    if agent.path and ((agent_new_x, agent_new_y) in [(other.x, other.y), (other_new_x, other_new_y)]):
                        # If we are in a rack and one of the agents is a picker we ignore clashses, assumed behaviour is Picker is loading
                        if not self._is_highway(agent_new_x, agent_new_y) and (agent.type == AgentType.PICKER or other.type == AgentType.PICKER) and agent.type != other.type:
                            # Allow Pickers to step over AGVs (if no other Picker at that shelf location) or AGVs to step over Pickers (if no other AGV at that shelf location)
                            if ((agent.type == AgentType.PICKER and self.grid[CollisionLayers.PICKERS, agent_new_y, agent_new_x] in [0, agent.id])
                                or (agent.type == AgentType.AGV and self.grid[CollisionLayers.AGVS, agent_new_y, agent_new_x] in [0, agent.id])):
                                commited_agents.add(agent.id)
                                continue
                        # If the agent's next action bumps it into another agent
                        if (agent_new_x, agent_new_y) == (other.x, other.y):
                            agent.req_action = Action.NOOP # Stop the action
                            # Check if the clash is not solved naturaly by the other agent moving away
                            if (other_new_x, other_new_y) in [(agent.x, agent.y), (agent_new_x, agent_new_y)] and not other.req_action in (Action.LEFT, Action.RIGHT):
                                if other.fixing_clash == 0:# If the others are not already fixing the clash
                                    clashes+=1
                                    agent.fixing_clash = _FIXING_CLASH_TIME # Agent start time for clash fixing
                                    new_path = self.find_path((agent.y, agent.x), (agent.path[-1][1] ,agent.path[-1][0]), agent)
                                    if new_path != []: # If the agent can find an alternative path, assign it if not let the other solve the clash
                                        agent.path = new_path
                                    else:
                                        agent.fixing_clash = 0
                        elif (agent_new_x, agent_new_y) == (other_new_x, other_new_y) and (agent_new_x, agent_new_y) != (agent.x, agent.y):
                            # If the agent's next action bumps it into another agent position after they take actions simultaneously
                            if agent.fixing_clash == 0 and other.fixing_clash == 0:
                                agent.req_action = Action.NOOP # If the agent's actions leads them in the position of another STOP
                                agent.fixing_clash = _FIXING_CLASH_TIME  # Agent wait one step while the other moves into place

        commited_agents = set([self.agents[id_ - 1] for id_ in commited_agents])
        failed_agents = set(agent_list) - commited_agents
        for agent in failed_agents:
            agent.req_action = Action.NOOP
        return clashes

    def resolve_stuck_agents(self) -> int:
        # This can happen when their goal is occupied after reaching their last step/re-calculating a path
        overall_stucks = 0
        moving_agents = [
            agent
            for agent in self.agents
            if agent.busy
            and agent.req_action not in (Action.LEFT, Action.RIGHT) # Don't count changing directions
            and (agent.req_action!=Action.TOGGLE_LOAD or (agent.x, agent.y) in self.goals) # Don't count loading or changing directions / if at goal
        ]
        for agent in moving_agents:
            agent_stuck_count = self.stuck_counters[agent.id - 1]
            agent_stuck_count.update((agent.x, agent.y))
            if _STUCK_THRESHOLD < agent_stuck_count.count < _STUCK_THRESHOLD + self.column_height + 2:  # Time to get out of aisle
                agent.req_action = Action.NOOP
                if agent.path:
                    new_path = self.find_path((agent.y, agent.x), (agent.path[-1][1], agent.path[-1][0]), agent)
                    # Picker should wait for AGV to arrive at destination regardless of stuck count
                    if new_path:
                        agent.path = new_path
                        if len(agent.path) == 1:
                            continue
                        agent_stuck_count.reset((agent.x, agent.y))
                        continue
                else:
                    overall_stucks += 1
                    agent.busy = False
                    agent_stuck_count.reset()
            if agent_stuck_count.count > _STUCK_THRESHOLD + self.column_height + 2:  # Time to get out of aisle
                overall_stucks += 1
                agent_stuck_count.reset((agent.x, agent.y))
                agent.req_action = Action.NOOP
                agent.busy = False
        return overall_stucks

    def _execute_forward(self, agent: Agent) -> None:
        agent.x, agent.y = agent.req_location(self.grid_size)
        agent.path = agent.path[1:]
        if agent.carrying_shelf:
            agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y
        if self.battery_drain:
            move_cost = self._movement_energy_cost(agent)
            agent.battery = max(0, agent.battery - move_cost)

    def _movement_energy_cost(self, agent: Agent) -> float:
        if agent.type != AgentType.AGV:
            return _BATTERY_CONSUMPTION_MOVE * self.picker_energy_factor
        if not agent.carrying_shelf:
            return _BATTERY_CONSUMPTION_MOVE_UNLOADED

        payload = float(max(0.0, getattr(agent.carrying_shelf, "weight", 0.0)))
        # Fallback preserves a higher loaded cost even if payload metadata is missing.
        if payload == 0.0:
            return _BATTERY_CONSUMPTION_MOVE * LOADED_FACTOR
        return _BATTERY_CONSUMPTION_MOVE * (1.0 + _AGV_WEIGHT_FACTOR * payload)

    def _execute_rotation(self, agent: Agent) -> None:
        agent.dir = agent.req_direction()

    def _collect_toggle_helpers(self, lead: Agent, radius: int = 1):
        """Gather AGV / picker helpers around `lead` for an atomic load/unload.

        AGV helpers require `req_action == TOGGLE_LOAD` and fall within Chebyshev
        `radius` of the lead so two AGVs can commit to a LARGE shelf without
        physically stacking on one cell. Pickers count as helpers by presence
        on the lead's cell (matching original retrieval semantics, in which the
        picker's macro-action stays NOOP while it waits beside the AGV).
        """
        agv_helpers: List[Agent] = []
        picker_helpers: List[Agent] = []
        for other in self.agents:
            if other.type == AgentType.AGV:
                # Count as helper if issuing TOGGLE_LOAD, OR if heading to the
                # same target as lead while stuck adjacent (enables LARGE 2-AGV
                # coordination without requiring same-cell occupancy).
                # NOTE: do NOT require `other.busy`. A committed LARGE helper that
                # has reached a cell adjacent to the (lead-occupied) shelf gets
                # its path/busy cleared by the env's stuck-resolution, yet it is
                # still part of the coalition (same target, not carrying). The
                # old `busy` requirement dropped it, leaving every LARGE load
                # permanently one AGV short (0 LARGE deliveries).
                heading_same = (
                    other.target == lead.target
                    and not other.carrying_shelf
                )
                if other.req_action != Action.TOGGLE_LOAD and not heading_same:
                    continue
                if other is lead:
                    agv_helpers.append(other)
                    continue
                if other.carrying_shelf is not None:
                    continue
                if max(abs(other.x - lead.x), abs(other.y - lead.y)) <= radius:
                    agv_helpers.append(other)
            elif other.type == AgentType.PICKER:
                # STANDARD shelves still expect the picker to sit on the AGV's
                # cell (legacy semantics); for LARGE we allow Chebyshev-1 so
                # two pickers don't have to collide on one grid cell.
                if max(abs(other.x - lead.x), abs(other.y - lead.y)) <= radius:
                    picker_helpers.append(other)
        return agv_helpers, picker_helpers

    def _execute_load(self, agent: Agent) -> None:
        # Pickers do not initiate loads.
        if agent.type == AgentType.PICKER:
            return
        shelf_id = self.grid[CollisionLayers.SHELVES, agent.y, agent.x]
        if not shelf_id:
            agent.busy = False
            return

        # Prevent accidental loads: only load if agent intentionally targeted this cell.
        if agent.target != 0 and self.action_id_to_coords_map.get(agent.target) != (agent.y, agent.x):
            agent.busy = False
            return

        shelf = self.shelfs[shelf_id - 1]
        req_agvs, req_pickers = shelf.required_agvs, shelf.required_pickers

        agv_helpers, picker_helpers = self._collect_toggle_helpers(agent)
        # Deterministic lead: lowest-id AGV among the helper pool that is on-cell.
        on_cell_agvs = [a for a in agv_helpers if a.x == agent.x and a.y == agent.y]
        if not on_cell_agvs:
            # No on-cell AGV (shouldn't happen since `agent` is on-cell); abort.
            agent.busy = False
            return
        lead = min(on_cell_agvs, key=lambda a: a.id)
        if lead is not agent:
            # Defer: the load is performed when the lead's own turn is processed.
            return

        # Always include the lead; fill remaining slots from nearest helpers by id.
        extra_helpers = sorted(
            [a for a in agv_helpers if a is not lead], key=lambda a: a.id
        )
        if len(agv_helpers) < req_agvs or len(picker_helpers) < req_pickers:
            agent.busy = False
            return
        agv_participants = [lead] + extra_helpers[: req_agvs - 1]
        picker_participants = sorted(picker_helpers, key=lambda a: a.id)[:req_pickers]

        # Commit load.
        shelf.carrier_group = list(agv_participants)
        lead.carrying_shelf = shelf
        self.grid[CollisionLayers.SHELVES, lead.y, lead.x] = 0
        self.grid[CollisionLayers.CARRIED_SHELVES, lead.y, lead.x] = shelf_id
        for a in agv_participants:
            a.busy = False
            if self.battery_drain:
                a.battery = max(0, a.battery - _BATTERY_CONSUMPTION_LOAD)
        for p in picker_participants:
            if self.battery_drain:
                p.battery = max(0, p.battery - _PICKER_LIFT_ENERGY)
            p.total_lifts += 1
            p.busy = False

    def _execute_unload(self, agent: Agent) -> None:
        # Pickers never carry a shelf; only the lead AGV unloads.
        if agent.type == AgentType.PICKER:
            return
        if (agent.x, agent.y) in self.goals or (agent.x, agent.y) in self._charging_station_positions or self.grid[CollisionLayers.SHELVES, agent.y, agent.x] != 0:
            agent.busy = False
            return
        if self._is_highway(agent.x, agent.y) or agent.carrying_shelf is None:
            return

        shelf = agent.carrying_shelf
        # Picker-gated return: placing a shelf back in storage needs the same
        # picker crew that the pickup did — 1 picker for STANDARD, 2 for LARGE
        # (PACKAGE_REQUIREMENTS). SOLO returns drop off with no picker. The
        # coordinator dispatches the helper picker(s) to the slot cell; until
        # `required_pickers` of them are within Chebyshev-1, the drop is blocked
        # so the AGV cannot abandon a crewed package alone.
        _, picker_helpers = self._collect_toggle_helpers(agent)
        if len(picker_helpers) < shelf.required_pickers:
            return

        self.grid[CollisionLayers.SHELVES, agent.y, agent.x] = shelf.id
        self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = 0
        agent.carrying_shelf = None
        agent.busy = False
        agent.has_delivered = False
        if self.battery_drain:
            agent.battery = max(0, agent.battery - _BATTERY_CONSUMPTION_LOAD)

        for picker in sorted(picker_helpers, key=lambda p: p.id)[: shelf.required_pickers]:
            if self.battery_drain:
                picker.battery = max(0, picker.battery - _PICKER_LIFT_ENERGY)
            picker.total_lifts += 1
            picker.busy = False
        shelf.carrier_group = []

    def _execute_charge(self, agent: Agent) -> None:
        """Charge the agent when it is on a charging station."""
        if (agent.x, agent.y) in self._charging_station_positions:
            agent.battery = min(_BATTERY_FULL, agent.battery + _BATTERY_CHARGE_RATE)

    def execute_micro_actions(self) -> None:
        for agent in self.agents:
            if agent.req_action == Action.FORWARD:
                self._execute_forward(agent)
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                self._execute_rotation(agent)
            elif agent.req_action == Action.TOGGLE_LOAD:
                if not agent.carrying_shelf:
                    self._execute_load(agent)
                else:
                    self._execute_unload(agent)
            elif agent.req_action == Action.CHARGE:
                self._execute_charge(agent)

    def _assign_package(self, shelf: "Shelf") -> None:
        """(Re)sample a package type + weight onto an existing shelf object.

        Mirrors ``Shelf.__init__`` so a re-requested shelf behaves exactly like
        a freshly-arrived one. Safe to call only when the shelf is neither in
        ``request_queue`` nor carried by an agent.
        """
        package_type = self._sample_package_type()
        shelf.package_type = package_type
        shelf.required_agvs, shelf.required_pickers = PACKAGE_REQUIREMENTS[package_type]
        weight_bias = {PackageType.SOLO: 0.5, PackageType.STANDARD: 1.0,
                       PackageType.LARGE: 1.75}[package_type]
        shelf.weight = float(np.clip(
            np.random.normal(_PKG_WEIGHT_MU * weight_bias, _PKG_WEIGHT_SIGMA),
            _PKG_WEIGHT_MIN, _PKG_WEIGHT_MAX
        ))

    def _replenish_request(self) -> None:
        """Append one fresh request, keeping the queue at constant size.

        Picks a shelf that is currently neither requested nor carried, samples
        a new package onto it, and enqueues it. No-op if no shelf is available
        (only possible in degenerate tiny layouts).
        """
        requested_ids = {s.id for s in self.request_queue}
        carried_ids = {a.carrying_shelf.id for a in self.agents if a.carrying_shelf}
        available = [
            s for s in self.shelfs
            if s.id not in requested_ids and s.id not in carried_ids
        ]
        if not available:
            return
        shelf = available[int(np.random.choice(len(available)))]
        self._assign_package(shelf)
        self.request_queue.append(shelf)

    def process_shelf_deliveries(self) -> int:
        shelf_deliveries = 0
        self._last_deliveries_by_type = {p: 0 for p in PackageType}
        for y, x in self.goals:
            shelf_id = self.grid[CollisionLayers.CARRIED_SHELVES, x, y]
            if not shelf_id or self.shelfs[shelf_id - 1] not in self.request_queue:
                continue
            shelf = self.shelfs[shelf_id - 1]
            self.request_queue.remove(shelf)
            if self.replenish_requests:
                self._replenish_request()

            agent = self.agents[self.grid[CollisionLayers.AGVS, x, y] - 1]
            if not agent.has_delivered:
                agent.has_delivered = True
            shelf_deliveries += 1
            self._last_deliveries_by_type[shelf.package_type] += 1

        if shelf_deliveries:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1

        return shelf_deliveries

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Dict[str, Any]:
        # Reset counters
        Shelf.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0
        self._last_deliveries_by_type = {p: 0 for p in PackageType}

        # Set seed
        self.seed(seed)

        # Make the shelfs
        self.shelfs = [
            Shelf(x, y, package_type=self._sample_package_type())
            for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if not self._is_highway(x, y) and (x, y) not in [(s.x, s.y) for s in self.charging_stations]
            and (x, y) not in self.goals
        ]
        self._higway_locs = np.array([(y, x) for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            ) if self._is_highway(x, y)])

        # Spawn agents on higwahy locations
        agent_loc_ids = np.random.choice(
            np.arange(len(self._higway_locs)),
            size=self.num_agents,
            replace=False,
        )
        agent_locs = [self._higway_locs[agent_loc_ids, 0], self._higway_locs[agent_loc_ids, 1]]
        # and direction
        agent_dirs = np.random.choice([d for d in Direction], size=self.num_agents)
        self.agents = [
            Agent(
                x,
                y,
                dir_,
                agent_type=agent_type,
                battery=self._initial_battery_for_agent(idx, agent_type),
            )
            for idx, (y, x, dir_, agent_type) in enumerate(zip(*agent_locs, agent_dirs, self._agent_types))
        ]

        self.stuck_counters = [StuckCounter((agent.x, agent.y)) for agent in self.agents]
        self._recalc_grid()

        indices = np.random.choice(len(self.shelfs), size=self.request_queue_size, replace=False)
        self.request_queue = [self.shelfs[i] for i in indices]
        return {}

    def step(
        self, action
    ) -> Tuple[bool, Dict[str, Any]]:
        # Attribute macro actions to agents and resolve conflicts
        agvs_distance_travelled, pickers_distance_travelled = self.attribute_macro_actions(action)
        clashes_count = self.resolve_move_conflict(self.agents)
        # Restart agents if they are stuck at the same position
        stucks_count = self.resolve_stuck_agents()

        # Execute micro actions
        self.execute_micro_actions()
        # Process shelf deliveries
        shelf_deliveries = self.process_shelf_deliveries()

        self._recalc_grid()
        self._cur_steps += 1
        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ) or (self.max_steps and self._cur_steps >= self.max_steps) or len(self.request_queue) == 0:
            done = True
        else:
            done = False

        info = self._build_info(
            agvs_distance_travelled,
            pickers_distance_travelled,
            clashes_count,
            stucks_count,
            shelf_deliveries,
            action,
        )
        return done, info

    def _build_info(
        self,
        agvs_distance_travelled: int,
        pickers_distance_travelled: int,
        clashes_count: int,
        stucks_count:  int,
        shelf_deliveries: int,
        macro_actions: List[int],
    ) -> Dict[str, np.ndarray]:
        info = {}
        agvs_idle_time = sum([int(agent.req_action in (Action.NOOP, Action.TOGGLE_LOAD)) for agent in self.agents[:self.num_agvs]])
        pickers_idle_time = sum([int(agent.req_action in (Action.NOOP, Action.TOGGLE_LOAD)) for agent in self.agents[self.num_agvs:]])
        info["vehicles_busy"] = [agent.busy for agent in self.agents]
        info["shelf_deliveries"] = shelf_deliveries
        info["clashes"] = clashes_count
        info["stucks"] = stucks_count
        info["agvs_distance_travelled"] = agvs_distance_travelled
        info["pickers_distance_travelled"] = pickers_distance_travelled
        info["agvs_idle_time"] = agvs_idle_time
        info["pickers_idle_time"] = pickers_idle_time
        info["battery_levels"] = [agent.battery for agent in self.agents]
        info["actions"] = macro_actions
        info["pkg_weights"] = [
            agent.carrying_shelf.weight if agent.carrying_shelf else 0.0
            for agent in self.agents
        ]
        info["total_lifts"] = [agent.total_lifts for agent in self.agents]
        info["charging_states"] = [agent.charging for agent in self.agents]
        info["deliveries_by_pkg_type"] = {
            p.name: self._last_deliveries_by_type.get(p, 0) for p in PackageType
        }
        info["carrying_pkg_type"] = [
            agent.carrying_shelf.package_type.name if agent.carrying_shelf else None
            for agent in self.agents
        ]
        return info

    def _package_type_at_location(self, location_index: int) -> Optional[PackageType]:
        """Map an action-space location index (0-based, minus goals/NOOP) to
        the package type of the shelf sitting at that coordinate, or None if no
        shelf exists. Used by the action mask to distinguish SOLO / LARGE."""
        if location_index >= len(self.shelf_action_ids):
            return None  # charging-station slot
        action_id = self.shelf_action_ids[location_index]
        y, x = self.action_id_to_coords_map[action_id]
        shelf_id = self.grid[CollisionLayers.SHELVES, y, x]
        if shelf_id == 0:
            shelf_id = self.grid[CollisionLayers.CARRIED_SHELVES, y, x]
        if shelf_id == 0:
            return None
        return self.shelfs[shelf_id - 1].package_type

    def compute_valid_action_masks(self, pickers_to_agvs=True, block_conflicting_actions=True):
        requested_items = self.get_shelf_request_information()
        empty_items = self.get_empty_shelf_information()
        carrying_shelf_info = self.get_carrying_shelf_information()
        num_goal_actions = len(self.goals)
        num_location_actions = self.action_size - (1 + num_goal_actions)
        num_shelf_actions = len(self.shelf_action_ids)

        # Expand shelf-only vectors (requested/empty) to full location action width
        # by appending charging action slots. Charging targets are always selectable.
        requested_locations = np.ones(num_location_actions, dtype=int)
        requested_locations[:num_shelf_actions] = requested_items
        empty_locations = np.ones(num_location_actions, dtype=int)
        empty_locations[:num_shelf_actions] = empty_items

        targets_agvs = [target - len(self.goals) - 1 for target in self.targets_agvs if target > len(self.goals)]
        targets_pickers = [target - len(self.goals) - 1 for target in self.targets_pickers if target > len(self.goals)]

        # Classify AGV targets by package type so we can (a) allow LARGE-shelf
        # co-targeting and (b) keep pickers off SOLO targets.
        target_pkg = {t: self._package_type_at_location(t) for t in targets_agvs}
        large_targets = [t for t, p in target_pkg.items() if p == PackageType.LARGE]
        picker_relevant_targets = [
            t for t, p in target_pkg.items() if p in (PackageType.STANDARD, PackageType.LARGE)
        ]

        # Compute valid location list for AGVs
        valid_location_list_agvs = np.array([
            empty_locations if carrying_shelf else requested_locations for carrying_shelf in carrying_shelf_info
        ])
        # Compute valid location list for Pickers: only follow AGVs heading to
        # STANDARD / LARGE shelves (SOLO shelves don't need pickers).
        if pickers_to_agvs:
            valid_location_list_pickers = np.zeros(num_location_actions, dtype=int)
            if picker_relevant_targets:
                valid_location_list_pickers[picker_relevant_targets] = 1
        else:
            valid_location_list_pickers = requested_locations.copy()
        # Mask out conflicting actions for agents of the same type.
        # LARGE shelves may be co-targeted by up to 2 AGVs and 2 pickers — don't
        # block those slots even when block_conflicting_actions is on.
        if block_conflicting_actions:
            blockable_agv_targets = [t for t in targets_agvs if t not in large_targets]
            valid_location_list_agvs[:, blockable_agv_targets] = 0
            blockable_picker_targets = [t for t in targets_pickers if target_pkg.get(t) != PackageType.LARGE]
            valid_location_list_pickers[blockable_picker_targets] = 0
        valid_action_masks = np.ones((self.num_agents, self.action_size))
        valid_action_masks[:self.num_agvs,  1 + len(self.goals):] = valid_location_list_agvs
        valid_action_masks[:self.num_agvs,  1 : 1 + len(self.goals)] = np.repeat(np.expand_dims(np.array(carrying_shelf_info), 1), len(self.goals), axis=1)
        valid_action_masks[self.num_agvs:,  1 + len(self.goals):] = valid_location_list_pickers
        valid_action_masks[self.num_agvs:,  1 : 1 + len(self.goals)] = 0
        return valid_action_masks

    def render(self, mode="human"):
        if not self.renderer:
            from battery_tarware.rendering import Viewer
            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
