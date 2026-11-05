"""
Core Gymnasium environment for Battery-TA-RWARE/Tarware.

Warehouse builds the grid layout, action and observation spaces, AGV/picker
agents, shelves, goals, charging stations, request queues, and package mix. It
executes macro actions through A* paths and primitive movement, resolves
collisions and stuck agents, applies battery drain and fast charging, supports
SOLO / STANDARD / HEAVY / PICKER_SOLO / LARGE load coordination, tracks
outbound deliveries and inbound returns, produces rich step info, and exposes
valid-action masks for training.
"""

import random
from typing import Dict, List, Optional, Tuple, Any

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces
from tarware.astar import astar_path
from tarware.definitions import (Action, AgentType, Direction,
                                 RewardType, CollisionLayers,
                                 PackageType, PackageDirection,
                                 PACKAGE_REQUIREMENTS,
                                 PACKAGE_REWARD_SCALE)
from tarware.spaces import observation_map
from tarware.utils import find_sections, get_next_micro_action

_FIXING_CLASH_TIME = 4
_STUCK_THRESHOLD = 5

# ---------------------------------------------------------------------------
# Battery model  (1 step = 5 s real time)
#
#   Unloaded runtime  : 12 h = 43 200 s = 8 640 steps
#   Loaded runtime    : 8 h  = 28 800 s = 5 760 steps  (at max package weight 7 kg)
#   Loaded/unloaded ratio: 12/8 = 1.5  →  _AGV_WEIGHT_FACTOR = 0.5/7 ≈ 0.07143 per kg
#   Load/unload action: ~65 s ≈ 13 movement steps
#   Fast-charge time  : 30 min = 1 800 s = 360 steps
# ---------------------------------------------------------------------------
_BATTERY_FULL = 100
_BATTERY_CONSUMPTION_MOVE = 100 / 8640   # ≈ 0.01157/step  — unloaded movement
_BATTERY_CONSUMPTION_LOAD = 0.15         # one-time cost per load/unload action
_BATTERY_CHARGE_RATE      = 100 / 360   # ≈ 0.278/step    — 30-min fast charge
_BATTERY_EXP_K = 3.0        # Steepness of exponential battery reward curve
_BATTERY_EXP_SCALE = 0.002  # Per-step reward magnitude scale
KINETIC_CONSUMPTION = _BATTERY_CONSUMPTION_MOVE
LOADED_FACTOR = 1.5          # fallback loaded multiplier when package weight unknown
STANDBY_CONSUMPTION = _BATTERY_CONSUMPTION_MOVE * 0.1  # ≈ 10 % of move cost

# Package weight model
_PKG_WEIGHT_MU = 5.0       # Mean package weight (kg) for STANDARD
_PKG_WEIGHT_SIGMA = 2.0    # Package weight std dev (kg)
_PKG_WEIGHT_MIN = 0.5      # Minimum package weight (kg)
_PKG_WEIGHT_MAX = 7.0     # Maximum package weight (kg)
# At max weight (7 kg): loaded factor = 1 + 0.07143*7 = 1.5  →  8 h runtime
# At lighter loads the AGV runs proportionally longer (e.g. 5 kg → ~8.8 h)
_AGV_WEIGHT_FACTOR = 0.5 / _PKG_WEIGHT_MAX   # = 0.5/_PKG_WEIGHT_MAX per kg (≈0.07143)
_PICKER_LIFT_ENERGY = 0.15 # picker one-time energy per lift assist (same as load action)

class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.x = x
        self.y = y

class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int, dir_: Direction, agent_type: AgentType):
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
        self.battery = float(random.randint(20, _BATTERY_FULL))  # New: Battery level
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
        self.reward_scale = PACKAGE_REWARD_SCALE[package_type]
        # Carrier group: AGVs that participated in the atomic load (lead + helpers).
        # Empty until a successful TOGGLE_LOAD; cleared on unload.
        self.carrier_group: List["Agent"] = []
        # Pickers that assisted with this shelf's load/unload.
        self.picker_group: List["Agent"] = []
        # Original shelf slot position — used as the inbound return target.
        self.home_pos: Tuple[int, int] = (x, y)
        # Current delivery direction: OUT = shelf→goal, IN = goal→shelf.
        self.direction: PackageDirection = PackageDirection.OUT
        # Package weight sampled from truncated normal distribution, biased by size.
        type_weight_bias = {
            PackageType.SOLO:        0.6,
            PackageType.STANDARD:    1.0,
            PackageType.LARGE:       1.6,
            PackageType.HEAVY:       1.4,   # heavy but doable alone; picker assist helps
            PackageType.PICKER_SOLO: 0.3,   # light item handled by picker
        }.get(package_type, 1.0)
        self.weight = float(np.clip(
            np.random.normal(_PKG_WEIGHT_MU * type_weight_bias, _PKG_WEIGHT_SIGMA),
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

class Warehouse(gym.Env):

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
        reward_type: RewardType,
        normalised_coordinates: bool=False,
        observation_type: str = "global",
        package_distribution: Optional[Dict] = None,
        large_convoy_adjacency: int = 1,
        num_chargers: Optional[int] = None,
        battery_drain_multiplier: float = 1.0,
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
        :param reward_type: Specifies if agents are rewarded individually or globally
        :type reward_type: RewardType
        :param observation_type: Specifies type of observations
        :type normalised_coordinates: str
        :param normalised_coordinates: Specifies whether absolute coordinates should be normalised
            with respect to total warehouse size
        :type normalised_coordinates: bool
        """

        self.goals: List[Tuple[int, int]] = []

        self.num_agvs = num_agvs
        self.num_pickers = num_pickers
        self.num_agents = num_agvs + num_pickers
        # Scarce-charger / high-drain knobs for the resource-competition env.
        # num_chargers=None keeps the default of one charger per agent.
        self.num_chargers = int(num_chargers) if num_chargers is not None else self.num_agents
        self._battery_drain_multiplier = float(battery_drain_multiplier)

        self._make_layout_from_params(shelf_columns, shelf_rows, column_height)
        self._charging_station_positions: set = {(s.x, s.y) for s in self.charging_stations}
        self.shelf_action_ids = list(range(len(self.goals) + 1, len(self.action_id_to_coords_map) - len(self.charging_stations) + 1))
        # If no Pickers are generated, AGVs can perform picks independently
        if num_pickers > 0:
            self._agent_types = [AgentType.AGV for _ in range(num_agvs)] + [AgentType.PICKER for _ in range(num_pickers)]
        else:
            self._agent_types = [AgentType.AGENT for _ in range(self.num_agents)]

        self.max_inactivity_steps: Optional[int] = max_inactivity_steps
        self.large_convoy_adjacency: int = max(1, int(large_convoy_adjacency))
        self.reward_type = reward_type
        # Initialize inactive steps counter to 0 to avoid comparisons with None before reset()
        self._cur_inactive_steps = 0
        self._cur_steps = 0
        self.max_steps = max_steps

        self.action_size = len(self.action_id_to_coords_map) + 1
        self.action_space = spaces.Tuple(tuple(self.num_agents * [spaces.Discrete(self.action_size)]))

        self.observation_space_mapper = observation_map[observation_type](
            self.num_agvs,
            self.num_pickers,
            self.grid_size,
            len(self.action_id_to_coords_map)-len(self.goals),
            normalised_coordinates,
        )
        self.observation_space = spaces.Tuple(tuple(self.observation_space_mapper.ma_spaces))

        self.request_queue_size = request_queue_size
        self.request_queue = []   # outbound: shelf slot → goal dock
        self.inbound_queue: List = []  # inbound: goal dock → shelf slot
        self.rack_groups = find_sections(list([loc for loc in self.action_id_to_coords_map.values() if (loc[1], loc[0]) not in self.goals]))
        self.agents: List[Agent] = []
        self.stuck_counters = []
        self._last_depleted_agents: List[int] = [] # battery <= 0
        self._last_depletion_events: List[int] = []
        self._last_delivery_events: List[Dict[str, Any]] = []
        self._last_return_events: List[Dict[str, Any]] = []
        self._last_returns_by_type: Dict[Any, int] = {}
        self._last_large_load_events: List[Dict[str, Any]] = []
        self._step_returns: int = 0

        self._package_distribution = self._normalise_package_distribution(package_distribution)
        self.renderer = None

    @property
    def targets_agvs(self):
        return [agent.target for agent in self.agents[:self.num_agvs]]

    @property
    def targets_pickers(self):
        return [agent.target for agent in self.agents[self.num_agvs:]]

    def _is_depleted(self, agent: Agent) -> bool:
        return agent.battery <= 0

    def _is_at_charger(self, agent: Agent) -> bool:
        return (agent.x, agent.y) in self._charging_station_positions

    def _macro_action_coords(self, macro_action: int) -> Optional[Tuple[int, int]]:
        return self.action_id_to_coords_map.get(macro_action)

    def _immobilize_agent(self, agent: Agent, clear_target: bool = True) -> None:
        agent.req_action = Action.NOOP
        agent.busy = False
        agent.path = None
        agent.charging = False
        if clear_target:
            agent.target = 0

    def _apply_depletion_constraints(self) -> None:
        for agent in self.agents:
            if self._is_depleted(agent) and not self._is_at_charger(agent):
                self._immobilize_agent(agent)

    def _large_convoy_ready(self, lead: Agent) -> bool:
        shelf = lead.carrying_shelf
        if shelf is None or shelf.package_type != PackageType.LARGE:
            return True
        carriers = list(shelf.carrier_group or [lead])
        req_agvs, _req_pickers = shelf.required_agvs, shelf.required_pickers
        adj = self.large_convoy_adjacency
        if len(carriers) < req_agvs or lead not in carriers:
            return False
        for carrier in carriers:
            if self._is_depleted(carrier):
                return False
            if max(abs(carrier.x - lead.x), abs(carrier.y - lead.y)) > adj:
                return False
            if carrier is not lead and carrier.target != lead.target:
                return False
        return True

    def _large_convoy_goal_ready(self, lead: Agent) -> bool:
        if not self._large_convoy_ready(lead):
            return False
        shelf = lead.carrying_shelf
        if shelf is None or shelf.package_type != PackageType.LARGE:
            return True
        adj = self.large_convoy_adjacency
        return all(max(abs(carrier.x - lead.x), abs(carrier.y - lead.y)) <= adj for carrier in shelf.carrier_group)

    def _is_large_convoy_carrier(self, agent: Agent) -> bool:
        for carrier in self.agents:
            shelf = carrier.carrying_shelf
            if shelf is not None and shelf.package_type == PackageType.LARGE and agent in shelf.carrier_group:
                return True
        return False

    def _get_convoy_lead(self, agent: Agent) -> Optional[Agent]:
        """Return the lead AGV if `agent` is a non-lead carrier in a LARGE convoy, else None.

        The lead is the AGV that holds carrying_shelf; followers are in carrier_group
        but have carrying_shelf=None. Followers use leader-follower path planning:
        each step they re-path to the leader's current position rather than the task target.
        """
        for other in self.agents:
            shelf = other.carrying_shelf
            if (
                shelf is not None
                and shelf.package_type == PackageType.LARGE
                and agent in shelf.carrier_group
                and agent is not other
            ):
                return other  # `other` is the lead; `agent` is the follower
        return None

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

        # New: Define charging stations. Default is one per agent; the
        # competition env passes a smaller num_chargers to force contention.
        n_chg = min(self.num_chargers, self.grid_size[1])
        self.charging_stations = [ChargingStation(i, 0) for i in range(self.grid_size[1])][:n_chg]

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

    def find_path(self, start, goal: Tuple[int, int], agent, care_for_agents: bool = True, loaded: Optional[bool] = None) -> List[Tuple[int, int]]:
        """
        Constructs a path from start to goal using A* on an agent-specific obstacle grid.

        Unloaded AGVs can move beneath standing shelves (shelves are not obstacles for them).
        Loaded AGVs must use corridors and treat standing shelves as obstacles.
        Pickers are further restricted to highway lanes only (non-highway cells are impassable).

        If `care_for_agents` is True, other agents are also treated as obstacles, except agents
        of the cooperating type that are waiting at the target location.

        Parameters:
        - care_for_agents (bool): Whether to consider other agents in the grid.
        - agent (Agent): The agent for which the path is being calculated.
        - start (tuple): The starting coordinates (x, y) of the agent.
        - goal (tuple): The goal coordinates (x, y) for the agent.

        Returns:
        - List of tuples representing the path from start to goal, or an empty list if no path is found.
        """
        grid = np.zeros(self.grid_size)
        # Loaded AGVs must navigate around standing shelves (corridors only).
        # Unloaded AGVs can move beneath shelves, so shelves are not obstacles for them.
        # `loaded` overrides the carrying_shelf check (e.g. for LARGE convoy followers).
        is_loaded = agent.carrying_shelf is not None if loaded is None else loaded
        if is_loaded:
            grid += self.grid[CollisionLayers.SHELVES]
        if care_for_agents:
            grid += self.grid[CollisionLayers.AGVS]
            grid += self.grid[CollisionLayers.PICKERS]
        # Agents should start a path regardless if some others are waiting around the target location
        grid[goal[0], goal[1]] = 0

        # PICKER_SOLO retrieval / transport requires pickers to enter aisle
        # cells. Detect either condition: the picker is already carrying a
        # shelf (only PICKER_SOLO ever ends up on a picker), or the goal cell
        # itself hosts a PICKER_SOLO shelf.
        picker_solo_path = False
        if agent.type == AgentType.PICKER:
            if agent.carrying_shelf is not None:
                picker_solo_path = True
            else:
                goal_shelf_id = self.grid[CollisionLayers.SHELVES, goal[0], goal[1]]
                if goal_shelf_id:
                    goal_shelf = self.shelfs[goal_shelf_id - 1]
                    if goal_shelf.package_type == PackageType.PICKER_SOLO:
                        picker_solo_path = True

        if agent.type == AgentType.PICKER and not picker_solo_path:
            # Pickers can only travel through the highway, but can access goal locations
            grid += (1-self.highways)
            grid[goal[0], goal[1]] -= not self._is_highway(goal[1], goal[0])
            for i in range(self.grid_size[1]):
                grid[self.grid_size[0] - 1, i] = 1

        # Ban Pickers crossing through racks if adjacent target location is chosen and force them thake the long way around.
        start_fix = (0, 0)
        if agent.type == AgentType.PICKER and not picker_solo_path and ((not self._is_highway(start[1], start[0])) and goal[0] == start[0] and abs(goal[1] - start[1]) == 1):
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
        result = astar_path(grid, adjusted_start, goal, allow_diagonal=False) # returns None if cant find path
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
            if self._is_depleted(agent):
                if (
                    self._is_at_charger(agent)
                    and macro_action != 0
                    and self._macro_action_coords(macro_action) == (agent.y, agent.x)
                ):
                    agent.req_action = Action.CHARGE
                else:
                    self._immobilize_agent(agent)
                continue
            # Leader-follower path planning for LARGE convoy non-lead carriers.
            # The follower replans every step toward the leader's current cell,
            # treating shelves as obstacles (same constraint as the loaded leader).
            convoy_lead = self._get_convoy_lead(agent)
            if convoy_lead is not None:
                leader_pos = (convoy_lead.y, convoy_lead.x)
                if (agent.y, agent.x) != leader_pos:
                    agent.path = self.find_path(
                        (agent.y, agent.x), leader_pos, agent,
                        care_for_agents=True, loaded=True
                    )
                    if agent.path:
                        agent.busy = True
                        agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                        agvs_distance_travelled += int(agent.type == AgentType.AGV)
                    else:
                        agent.req_action = Action.NOOP
                else:
                    agent.req_action = Action.NOOP
                    agent.path = []
                continue

            # Collision avoidance logic
            if agent.fixing_clash > 0:
                agent.fixing_clash -= 1
            if not agent.busy:
                agent.target = 0
                if macro_action != 0:
                    target_coords = self.action_id_to_coords_map[macro_action]
                    if target_coords == (agent.y, agent.x) and (agent.x, agent.y) in self._charging_station_positions:
                        agent.target = macro_action
                        agent.req_action = Action.CHARGE
                        agent.busy = False
                        agent.charging = False
                        continue
                    agent.path = self.find_path((agent.y, agent.x), target_coords, agent, care_for_agents=False)
                    if agent.path:
                        agent.busy = True
                        agent.target = macro_action
                        agent.req_action = get_next_micro_action(agent.x, agent.y, agent.dir, agent.path[0])
                        self.stuck_counters[agent.id - 1].reset((agent.x, agent.y))
                    elif agent.type in (AgentType.AGV, AgentType.AGENT):
                        # Already at target cell: keep issuing TOGGLE_LOAD each step
                        # until the load requirements are met (needed for LARGE tasks
                        # where helpers may arrive later).
                        agent.target = macro_action
                        agent.req_action = Action.TOGGLE_LOAD
            else:
                # Check if agent finished the given path, if not continue the path
                if agent.path is None or agent.path == []:
                    if agent.type in [AgentType.AGV, AgentType.AGENT]:
                        agent.req_action = Action.TOGGLE_LOAD
                    elif agent.type == AgentType.PICKER:
                        # PICKER_SOLO leads: issue TOGGLE_LOAD at the shelf cell
                        # (to load) or at the carried shelf's home position (to
                        # unload inbound). All other pickers (STANDARD/HEAVY/
                        # LARGE assist) wait for the AGV by setting busy=False.
                        picker_solo_action = False
                        if agent.carrying_shelf is not None:
                            shelf = agent.carrying_shelf
                            if (
                                shelf.required_agvs == 0
                                and (agent.x, agent.y) == shelf.home_pos
                            ):
                                picker_solo_action = True
                        else:
                            shelf_id = self.grid[CollisionLayers.SHELVES, agent.y, agent.x]
                            if shelf_id:
                                shelf_here = self.shelfs[shelf_id - 1]
                                if shelf_here.package_type == PackageType.PICKER_SOLO:
                                    picker_solo_action = True
                        if picker_solo_action:
                            agent.req_action = Action.TOGGLE_LOAD
                        else:
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
                    # Pickers about to land on the final cell: AGV-assist pickers
                    # wait for the AGV's TOGGLE_LOAD, but PICKER_SOLO leads and
                    # inbound-return pickers proceed without an AGV.
                    if agent.type == AgentType.PICKER:
                        final_y, final_x = agent.path[-1][1], agent.path[-1][0]
                        picker_independent = False
                        if agent.carrying_shelf is not None:
                            picker_independent = agent.carrying_shelf.required_agvs == 0
                        else:
                            shelf_id_final = self.grid[CollisionLayers.SHELVES, final_y, final_x]
                            if shelf_id_final:
                                shelf_final = self.shelfs[shelf_id_final - 1]
                                if shelf_final.package_type == PackageType.PICKER_SOLO:
                                    picker_independent = True
                        if not picker_independent:
                            if (
                                self.grid[CollisionLayers.AGVS, final_y, final_x] == 0
                                or self.agents[self.grid[CollisionLayers.AGVS, final_y, final_x] - 1].req_action != Action.TOGGLE_LOAD
                            ):
                                agent.req_action = Action.NOOP
                            elif (
                                self.grid[CollisionLayers.AGVS, final_y, final_x] != 0
                                and self.agents[self.grid[CollisionLayers.AGVS, final_y, final_x] - 1].req_action == Action.TOGGLE_LOAD
                            ):
                                self.stuck_counters[agent.id - 1].reset((agent.x, agent.y))

            # Check if agent has finished its path and should charge
            if (
                (agent.path is None or agent.path == [])
                and (agent.charging or (agent.x, agent.y) in self._charging_station_positions)
                and agent.battery < _BATTERY_FULL
                and (agent.x, agent.y) in self._charging_station_positions
            ):
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
                        # In a shelf aisle, AGVs and Pickers occupy separate grid layers and do not block each other.
                        if not self._is_highway(agent_new_x, agent_new_y) and (agent.type == AgentType.PICKER or other.type == AgentType.PICKER) and agent.type != other.type:
                            # Allow same-type cell occupancy only when the cell is free of other same-type agents.
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
        if self._is_depleted(agent):
            self._immobilize_agent(agent)
            return
        agent.x, agent.y = agent.req_location(self.grid_size)
        agent.path = agent.path[1:]
        if agent.carrying_shelf:
            agent.carrying_shelf.x, agent.carrying_shelf.y = agent.x, agent.y
        move_cost = self._movement_energy_cost(agent)
        agent.battery = max(0, agent.battery - move_cost)

    def _movement_energy_cost(self, agent: Agent) -> float:
        m = self._battery_drain_multiplier
        if agent.type not in [AgentType.AGV, AgentType.AGENT]:
            return _BATTERY_CONSUMPTION_MOVE * m
        if not agent.carrying_shelf:
            return _BATTERY_CONSUMPTION_MOVE * m

        payload = float(max(0.0, getattr(agent.carrying_shelf, "weight", 0.0)))
        # Fallback preserves a higher loaded cost even if payload metadata is missing.
        if payload == 0.0:
            return _BATTERY_CONSUMPTION_MOVE * LOADED_FACTOR * m
        return _BATTERY_CONSUMPTION_MOVE * (1.0 + _AGV_WEIGHT_FACTOR * payload) * m

    def _execute_rotation(self, agent: Agent) -> None:
        if self._is_depleted(agent):
            self._immobilize_agent(agent)
            return
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
            if other.type in (AgentType.AGV, AgentType.AGENT):
                # Count as helper if issuing TOGGLE_LOAD, OR if heading to the
                # same target as lead while stuck adjacent (enables LARGE 2-AGV
                # coordination without requiring same-cell occupancy).
                heading_same = (
                    other.target == lead.target
                    and other.busy
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

    def _execute_load(self, agent: Agent, rewards: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        if self._is_depleted(agent):
            return rewards
        shelf_id = self.grid[CollisionLayers.SHELVES, agent.y, agent.x]
        if not shelf_id:
            agent.busy = False
            return rewards

        shelf = self.shelfs[shelf_id - 1]

        # Picker-led shelves (PICKER_SOLO, required_agvs == 0) are loaded by
        # pickers. AGV-led shelves (everything else) are loaded by AGVs.
        picker_led = shelf.required_agvs == 0
        if picker_led and agent.type != AgentType.PICKER:
            agent.busy = False
            return rewards
        if not picker_led and agent.type == AgentType.PICKER:
            return rewards

        # At goal cells only inbound shelves may be picked up (the shelf was
        # dropped here by process_shelf_deliveries and awaits restock).
        if (agent.x, agent.y) in self.goals:
            if shelf not in self.inbound_queue:
                agent.busy = False
                return rewards
            # Allow load to proceed below (skip the normal highway guard).
        elif self._is_highway(agent.x, agent.y):
            # Mid-highway load not allowed (unchanged guard from original code).
            agent.busy = False
            return rewards

        # Prevent accidental loads: only load if agent intentionally targeted this cell.
        if agent.target != 0 and self.action_id_to_coords_map.get(agent.target) != (agent.y, agent.x):
            agent.busy = False
            return rewards

        req_agvs = shelf.required_agvs
        # At the goal dock, AGV picks up the inbound package alone — pickers
        # assist with rack placement (unload), not dock pickup (load). The
        # picker-led PICKER_SOLO case always uses the carrier picker (req=1).
        if picker_led:
            req_pickers = shelf.required_pickers
        else:
            req_pickers = 0 if (agent.x, agent.y) in self.goals else shelf.required_pickers

        agv_helpers, picker_helpers = self._collect_toggle_helpers(agent)

        if picker_led:
            on_cell_pickers = [p for p in picker_helpers if p.x == agent.x and p.y == agent.y]
            if not on_cell_pickers:
                agent.busy = False
                return rewards
            lead = min(on_cell_pickers, key=lambda p: p.id)
            if lead is not agent:
                return rewards
            if len(picker_helpers) < req_pickers:
                agent.busy = False
                return rewards
            picker_participants = sorted(picker_helpers, key=lambda p: p.id)[:req_pickers]
            if any(self._is_depleted(p) for p in picker_participants):
                agent.busy = False
                return rewards
            agv_participants: List[Agent] = []
        else:
            # Deterministic lead: lowest-id AGV among the helper pool that is on-cell.
            on_cell_agvs = [a for a in agv_helpers if a.x == agent.x and a.y == agent.y]
            if not on_cell_agvs:
                # No on-cell AGV (shouldn't happen since `agent` is on-cell); abort.
                agent.busy = False
                return rewards
            lead = min(on_cell_agvs, key=lambda a: a.id)
            if lead is not agent:
                # Defer: the load is performed when the lead's own turn is processed.
                return rewards

            # Always include the lead; fill remaining slots from nearest helpers by id.
            extra_helpers = sorted(
                [a for a in agv_helpers if a is not lead], key=lambda a: a.id
            )
            if len(agv_helpers) < req_agvs or len(picker_helpers) < req_pickers:
                agent.busy = False
                return rewards
            agv_participants = [lead] + extra_helpers[: req_agvs - 1]
            picker_participants = sorted(picker_helpers, key=lambda a: a.id)[:req_pickers]
            if any(self._is_depleted(a) for a in agv_participants) or any(self._is_depleted(p) for p in picker_participants):
                agent.busy = False
                return rewards

        # Commit load.
        shelf.carrier_group = list(agv_participants)
        shelf.picker_group = list(picker_participants)
        if shelf.package_type == PackageType.LARGE:
            self._last_large_load_events.append(
                {
                    "shelf_id": int(shelf.id),
                    "package_type": shelf.package_type.name,
                    "carrier_ids": [int(a.id - 1) for a in agv_participants],
                    "picker_ids": [int(p.id - 1) for p in picker_participants],
                    "position": [int(shelf.x), int(shelf.y)],
                }
            )
        lead.carrying_shelf = shelf
        self.grid[CollisionLayers.SHELVES, lead.y, lead.x] = 0
        self.grid[CollisionLayers.CARRIED_SHELVES, lead.y, lead.x] = shelf_id
        for a in agv_participants:
            a.busy = False
            a.battery = max(0, a.battery - _BATTERY_CONSUMPTION_LOAD)
        for p in picker_participants:
            p.battery = max(0, p.battery - _PICKER_LIFT_ENERGY)
            p.total_lifts += 1
            p.busy = False

        # Load assist reward: small signal to incentivise picking up the shelf.
        # Pickers and AGVs get equal share of 0.5×scale so pickers are not
        # disadvantaged relative to SOLO AGVs (breaks SOLO dominance).
        scale = shelf.reward_scale
        if self.reward_type == RewardType.GLOBAL:
            rewards += 0.5 * scale * max(1, req_pickers)
        elif self.reward_type == RewardType.INDIVIDUAL:
            all_participants = list(agv_participants) + list(picker_participants)
            if all_participants:
                share = (0.5 * scale) / len(all_participants)
                for p in all_participants:
                    rewards[p.id - 1] += share
        return rewards

    def _execute_unload(self, agent: Agent, rewards: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        if self._is_depleted(agent) or agent.carrying_shelf is None:
            return rewards

        shelf = agent.carrying_shelf
        picker_led = shelf.required_agvs == 0

        # --- Inbound restock: carrier returning shelf to its home slot ---
        if shelf.direction == PackageDirection.IN:
            if (agent.x, agent.y) != shelf.home_pos:
                # Wrong position — block unload until carrier reaches home slot.
                return rewards
            # Require picker assists at unload for STANDARD / HEAVY.
            _, picker_helpers = self._collect_toggle_helpers(agent)
            if shelf.package_type in (PackageType.STANDARD, PackageType.HEAVY):
                if len(picker_helpers) < shelf.required_pickers:
                    return rewards
            # Drop shelf at home position.
            self.grid[CollisionLayers.SHELVES, agent.y, agent.x] = shelf.id
            self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = 0
            agent.carrying_shelf = None
            agent.busy = False
            if picker_led:
                agent.battery = max(0, agent.battery - _PICKER_LIFT_ENERGY)
            else:
                agent.battery = max(0, agent.battery - _BATTERY_CONSUMPTION_LOAD)
            shelf.direction = PackageDirection.OUT
            if shelf in self.inbound_queue:
                self.inbound_queue.remove(shelf)
            scale = shelf.reward_scale
            # Energy cost for assist pickers (not the picker-led carrier itself).
            if shelf.package_type in (PackageType.STANDARD, PackageType.HEAVY):
                for picker in picker_helpers[: shelf.required_pickers]:
                    picker.battery = max(0, picker.battery - _PICKER_LIFT_ENERGY)
                    picker.total_lifts += 1
                    picker.busy = False
            # Inbound delivery reward (same scale as outbound).
            if self.reward_type == RewardType.GLOBAL:
                rewards += 1.0 * scale
            elif self.reward_type == RewardType.INDIVIDUAL:
                all_participants = (shelf.carrier_group or [agent]) + shelf.picker_group
                share = (1.0 * scale) / len(all_participants)
                for p in all_participants:
                    rewards[p.id - 1] += share
            if picker_led:
                carrier_ids: List[int] = []
                picker_ids = [int(p.id - 1) for p in (shelf.picker_group or [agent])]
            else:
                carrier_ids = [int(a.id - 1) for a in (shelf.carrier_group or [agent])]
                picker_ids = [int(p.id - 1) for p in shelf.picker_group]
            self._last_return_events.append(
                {
                    "shelf_id": int(shelf.id),
                    "package_type": shelf.package_type.name,
                    "carrier_ids": carrier_ids,
                    "picker_ids": picker_ids,
                    "home_pos": [int(shelf.home_pos[0]), int(shelf.home_pos[1])],
                    "count": 1,
                }
            )
            self._last_returns_by_type[shelf.package_type] = self._last_returns_by_type.get(shelf.package_type, 0) + 1
            self._step_returns += 1
            shelf.carrier_group = []
            shelf.picker_group = []
            # Inbound restock counts as activity (prevents premature inactivity timeout).
            self._cur_inactive_steps = 0
            return rewards

        # --- Outbound: block drop at goals (handled by process_shelf_deliveries),
        #     chargers, highways, and occupied slots. ---
        if (agent.x, agent.y) in self.goals or (agent.x, agent.y) in self._charging_station_positions or self.grid[CollisionLayers.SHELVES, agent.y, agent.x] != 0:
            agent.busy = False
            return rewards
        if self._is_highway(agent.x, agent.y):
            return rewards

        if shelf.package_type == PackageType.LARGE and not self._large_convoy_ready(agent):
            return rewards
        req_pickers = shelf.required_pickers
        _, picker_helpers = self._collect_toggle_helpers(agent)
        if shelf.package_type in (PackageType.STANDARD, PackageType.HEAVY):
            if len(picker_helpers) < req_pickers:
                return rewards

        self.grid[CollisionLayers.SHELVES, agent.y, agent.x] = shelf.id
        self.grid[CollisionLayers.CARRIED_SHELVES, agent.y, agent.x] = 0
        agent.carrying_shelf = None
        agent.busy = False
        agent.has_delivered = False
        if picker_led:
            agent.battery = max(0, agent.battery - _PICKER_LIFT_ENERGY)
        else:
            agent.battery = max(0, agent.battery - _BATTERY_CONSUMPTION_LOAD)

        scale = shelf.reward_scale
        if shelf.package_type in (PackageType.STANDARD, PackageType.HEAVY):
            for picker in picker_helpers[: req_pickers]:
                picker.battery = max(0, picker.battery - _PICKER_LIFT_ENERGY)
                picker.total_lifts += 1
                picker.busy = False

        if self.reward_type == RewardType.GLOBAL:
            rewards += 0.5 * scale * max(1, req_pickers)
        elif self.reward_type == RewardType.INDIVIDUAL:
            if picker_led:
                rewards[agent.id - 1] += 0.1 * scale
            elif req_pickers == 0:
                rewards[agent.id - 1] += 0.1 * scale
            elif picker_helpers:
                rewards[picker_helpers[0].id - 1] += 0.1 * scale
        shelf.carrier_group = []
        shelf.picker_group = []
        return rewards

    def _execute_charge(self, agent: Agent) -> float:
        """Charge the agent and return a small reward proportional to energy gained."""
        if (agent.x, agent.y) in self._charging_station_positions:
            before = agent.battery
            agent.battery = min(_BATTERY_FULL, agent.battery + _BATTERY_CHARGE_RATE)
            return (agent.battery - before) * 0.001   # +0.005 per full charge step
        return 0.0

    def execute_micro_actions(self, rewards: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        for agent in self.agents:
            if self._is_depleted(agent) and agent.req_action != Action.CHARGE:
                self._immobilize_agent(agent)
                continue
            if agent.req_action == Action.FORWARD:
                self._execute_forward(agent)
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                self._execute_rotation(agent)
            elif agent.req_action == Action.TOGGLE_LOAD:
                if not agent.carrying_shelf:
                    rewards = self._execute_load(agent, rewards)
                else:
                    rewards = self._execute_unload(agent, rewards)
            elif agent.req_action == Action.CHARGE:
                rewards[agent.id - 1] += self._execute_charge(agent)
        return rewards

    def _replenish_outbound(self) -> None:
        """Sample a new outbound request when the queue shrinks after a delivery."""
        occupied = set(s.id for s in self.request_queue) | set(s.id for s in self.inbound_queue)
        candidates = [
            s for s in self.shelfs
            if s.id not in occupied
            and self.grid[CollisionLayers.SHELVES, s.y, s.x] == s.id  # still at home slot
        ]
        if candidates:
            new_shelf = self.np_random.choice(candidates)
            new_shelf.direction = PackageDirection.OUT
            self.request_queue.append(new_shelf)

    def process_shelf_deliveries(self, rewards: np.ndarray[Any, np.dtype[np.float64]]) -> Tuple[np.ndarray[Any, np.dtype[np.float64]], int]:
        shelf_deliveries = 0
        self._last_deliveries_by_type = {p: 0 for p in PackageType}
        self._last_delivery_events = []
        for goal_x, goal_y in self.goals:
            shelf_id = self.grid[CollisionLayers.CARRIED_SHELVES, goal_y, goal_x]
            if not shelf_id or self.shelfs[shelf_id - 1] not in self.request_queue:
                continue
            shelf = self.shelfs[shelf_id - 1]
            # The carrier may be an AGV (default) or a picker (PICKER_SOLO).
            carrier_grid_id = self.grid[CollisionLayers.AGVS, goal_y, goal_x]
            if not carrier_grid_id:
                carrier_grid_id = self.grid[CollisionLayers.PICKERS, goal_y, goal_x]
            if not carrier_grid_id:
                continue
            agent = self.agents[carrier_grid_id - 1]
            if self._is_depleted(agent) or agent.carrying_shelf is not shelf:
                continue
            if shelf.package_type == PackageType.LARGE and not self._large_convoy_goal_ready(agent):
                continue
            if not agent.has_delivered:
                self.request_queue.remove(shelf)
                scale = shelf.reward_scale
                agent.has_delivered = True
                picker_led = shelf.required_agvs == 0
                # Reward: split across carrier_group AND picker_group.
                if self.reward_type == RewardType.GLOBAL:
                    rewards += 1.0 * scale
                elif self.reward_type == RewardType.INDIVIDUAL:
                    if picker_led:
                        all_participants = shelf.picker_group or [agent]
                    else:
                        all_participants = (shelf.carrier_group or [agent]) + shelf.picker_group
                    share = (1.0 * scale) / len(all_participants)
                    for p in all_participants:
                        rewards[p.id - 1] += share

                if picker_led:
                    carrier_ids: List[int] = []
                    picker_ids = [int(p.id - 1) for p in (shelf.picker_group or [agent])]
                else:
                    carrier_ids = [int(a.id - 1) for a in (shelf.carrier_group or [agent])]
                    picker_ids = [int(p.id - 1) for p in shelf.picker_group]
                # --- Drop shelf at goal dock; start inbound cycle ---
                self.grid[CollisionLayers.SHELVES, goal_y, goal_x] = shelf.id
                self.grid[CollisionLayers.CARRIED_SHELVES, goal_y, goal_x] = 0
                agent.carrying_shelf = None
                agent.has_delivered = False
                agent.busy = False
                shelf.direction = PackageDirection.IN
                shelf.carrier_group = []
                shelf.picker_group = []
                self.inbound_queue.append(shelf)
                # Replenish outbound queue to keep throughput steady.
                self._replenish_outbound()

                shelf_deliveries += 1
                self._last_deliveries_by_type[shelf.package_type] += 1
                self._last_delivery_events.append(
                    {
                        "shelf_id": int(shelf.id),
                        "package_type": shelf.package_type.name,
                        "carrier_ids": carrier_ids,
                        "picker_ids": picker_ids,
                        "goal": [int(goal_x), int(goal_y)],
                        "count": 1,
                    }
                )

        if shelf_deliveries:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1

        return rewards, shelf_deliveries

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple:
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
            Agent(x, y, dir_, agent_type = agent_type)
            for y, x, dir_, agent_type in zip(*agent_locs, agent_dirs, self._agent_types)
        ]

        self.stuck_counters = [StuckCounter((agent.x, agent.y)) for agent in self.agents]
        self._recalc_grid()

        indices = np.random.choice(len(self.shelfs), size=self.request_queue_size, replace=False)
        self.request_queue = [self.shelfs[i] for i in indices]
        for shelf in self.request_queue:
            shelf.direction = PackageDirection.OUT
        self.inbound_queue = []
        self.observation_space_mapper.extract_environment_info(self)
        obs = tuple([self.observation_space_mapper.observation(agent) for agent in self.agents])
        return obs, {}

    def step(
        self, action
    ) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
        # Attribute macro actions to agents and resolve conflicts
        prev_battery = [agent.battery for agent in self.agents]
        self._last_delivery_events = []
        self._last_return_events = []
        self._last_returns_by_type = {p: 0 for p in PackageType}
        self._last_large_load_events = []
        self._step_returns = 0
        self._apply_depletion_constraints()
        agvs_distance_travelled, pickers_distance_travelled = self.attribute_macro_actions(action)
        clashes_count = self.resolve_move_conflict(self.agents)
        # Restart agents if they are stuck at the same position
        stucks_count = self.resolve_stuck_agents()
        self._apply_depletion_constraints()

        rewards = np.zeros(self.num_agents)
        # Apply penalty for inactivity
        rewards -= 0.001
        # Exponential battery reward: smoothly negative when low, positive when high.
        # Centred at b=50 (r≈0), shaped by exp so extremes are increasingly punished/rewarded.
        # r(b) = scale * (exp(k * b/100) − exp(k * 0.5))
        # At b=0: ≈ −0.007  |  b=50: 0  |  b=100: ≈ +0.031  (with k=3, scale=0.002)
        for i, agent in enumerate(self.agents):
            b_norm = agent.battery / _BATTERY_FULL
            rewards[i] += _BATTERY_EXP_SCALE * (
                np.exp(_BATTERY_EXP_K * b_norm) - np.exp(_BATTERY_EXP_K * 0.5)
            )
        # Execute micro actions
        rewards = self.execute_micro_actions(rewards)
        # Process shelf deliveries
        rewards, shelf_deliveries = self.process_shelf_deliveries(rewards)
        self._last_depleted_agents = [i for i, agent in enumerate(self.agents) if self._is_depleted(agent)]
        self._last_depletion_events = [
            i for i, (before, agent) in enumerate(zip(prev_battery, self.agents))
            if before > 0 and self._is_depleted(agent)
        ]

        self._recalc_grid()
        self._cur_steps += 1
        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ) or (self.max_steps and self._cur_steps >= self.max_steps):
            terminateds = truncateds = self.num_agents * [True]
        else:
            terminateds = truncateds =  self.num_agents * [False]

        self.observation_space_mapper.extract_environment_info(self)
        new_obs = tuple([self.observation_space_mapper.observation(agent) for agent in self.agents])
        info = self._build_info(
            agvs_distance_travelled,
            pickers_distance_travelled,
            clashes_count,
            stucks_count,
            shelf_deliveries,
            action,
        )
        return new_obs, list(rewards), terminateds, truncateds, info

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
        info["agent_positions"] = [[agent.x, agent.y] for agent in self.agents]
        info["agent_directions"] = [agent.dir.name for agent in self.agents]
        info["actions"] = macro_actions
        info["depleted_agents"] = list(self._last_depleted_agents)
        info["depletion_events"] = list(self._last_depletion_events)
        info["delivery_events"] = list(self._last_delivery_events)
        info["shelf_returns"] = int(self._step_returns)
        info["return_events"] = list(self._last_return_events)
        info["returns_by_pkg_type"] = {
            p.name: self._last_returns_by_type.get(p, 0) for p in PackageType
        }
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
        info["large_load_events"] = list(self._last_large_load_events)
        info["pending_large_positions"] = [
            [int(shelf.x), int(shelf.y)]
            for shelf in self.request_queue
            if shelf.package_type == PackageType.LARGE
            and self.grid[CollisionLayers.CARRIED_SHELVES, shelf.y, shelf.x] == 0
        ]
        info["request_queue_ids"] = [int(shelf.id) for shelf in self.request_queue]
        return info

    def _package_type_at_location(self, location_index: int) -> Optional[PackageType]:
        """Map an action-space location index (0-based, minus goals/NOOP) to
        the package type of the shelf sitting at that coordinate, or None if no
        shelf exists. Used by the action mask to distinguish package types."""
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

    def _package_type_at_coord(self, y: int, x: int) -> Optional[PackageType]:
        """Resolve the package type of any shelf currently sitting at (y, x)."""
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

        # Classify ALL shelf-bearing location indices by package type so we can
        # gate AGV vs picker targeting by capability:
        #   - AGVs cannot target PICKER_SOLO (req_agvs == 0)
        #   - Pickers should only target shelves they actually assist with
        #     (STANDARD, HEAVY, LARGE) or carry alone (PICKER_SOLO)
        all_shelf_pkg = {t: self._package_type_at_location(t) for t in range(num_shelf_actions)}
        large_targets = [t for t, p in all_shelf_pkg.items() if p == PackageType.LARGE]
        picker_solo_locations = [t for t, p in all_shelf_pkg.items() if p == PackageType.PICKER_SOLO]
        target_pkg_agv = {t: all_shelf_pkg.get(t) for t in targets_agvs}
        picker_relevant_targets = [
            t for t, p in target_pkg_agv.items()
            if p in (PackageType.STANDARD, PackageType.HEAVY, PackageType.LARGE)
        ]

        # Compute valid location list for AGVs (with PICKER_SOLO shelves masked).
        valid_location_list_agvs = np.array([
            empty_locations if carrying_shelf else requested_locations for carrying_shelf in carrying_shelf_info
        ])
        if picker_solo_locations:
            valid_location_list_agvs[:, picker_solo_locations] = 0
        # Compute valid location list for Pickers: follow AGVs heading to
        # STANDARD / HEAVY / LARGE shelves, plus PICKER_SOLO shelves the
        # picker can serve alone.
        if pickers_to_agvs:
            valid_location_list_pickers = np.zeros(num_location_actions, dtype=int)
            if picker_relevant_targets:
                valid_location_list_pickers[picker_relevant_targets] = 1
            if picker_solo_locations:
                # Only PICKER_SOLO shelves still in the outbound request queue
                # (or carried/inbound, handled via requested_locations) should
                # be selectable.
                solo_mask = requested_locations.copy()
                solo_indicator = np.zeros(num_location_actions, dtype=int)
                solo_indicator[picker_solo_locations] = 1
                valid_location_list_pickers = np.maximum(
                    valid_location_list_pickers, solo_mask * solo_indicator
                )
        else:
            valid_location_list_pickers = requested_locations.copy()
        # Mask out conflicting actions for agents of the same type.
        # LARGE shelves may be co-targeted by up to 2 AGVs and 2 pickers — don't
        # block those slots even when block_conflicting_actions is on.
        if block_conflicting_actions:
            blockable_agv_targets = [t for t in targets_agvs if t not in large_targets]
            valid_location_list_agvs[:, blockable_agv_targets] = 0
            # HEAVY and LARGE allow multiple pickers on the same shelf target.
            picker_coop_types = (PackageType.LARGE, PackageType.HEAVY)
            blockable_picker_targets = [
                t for t in targets_pickers
                if all_shelf_pkg.get(t) not in picker_coop_types
            ]
            valid_location_list_pickers[blockable_picker_targets] = 0
        valid_action_masks = np.ones((self.num_agents, self.action_size))
        valid_action_masks[:self.num_agvs,  1 + len(self.goals):] = valid_location_list_agvs
        goal_eligible_agvs = [
            bool(carrying_shelf_info[i]) or self._is_large_convoy_carrier(self.agents[i])
            for i in range(self.num_agvs)
        ]
        valid_action_masks[:self.num_agvs,  1 : 1 + len(self.goals)] = np.repeat(np.expand_dims(np.array(goal_eligible_agvs), 1), len(self.goals), axis=1)
        valid_action_masks[self.num_agvs:,  1 + len(self.goals):] = valid_location_list_pickers
        # Pickers carrying a PICKER_SOLO shelf may select a goal action to
        # deliver; all other pickers are blocked from goal actions.
        picker_goal_eligible = np.array([
            bool(self.agents[self.num_agvs + j].carrying_shelf)
            for j in range(self.num_agents - self.num_agvs)
        ])
        valid_action_masks[self.num_agvs:,  1 : 1 + len(self.goals)] = np.repeat(
            np.expand_dims(picker_goal_eligible.astype(int), 1), len(self.goals), axis=1
        )
        for i, agent in enumerate(self.agents):
            if self._is_depleted(agent):
                valid_action_masks[i, :] = 0
                valid_action_masks[i, 0] = 1
                if self._is_at_charger(agent):
                    for action_id, coords in self.action_id_to_coords_map.items():
                        if coords == (agent.y, agent.x):
                            valid_action_masks[i, action_id] = 1
            elif np.any(valid_action_masks[i, 1:] > 0):
                valid_action_masks[i, 0] = 0
        return valid_action_masks

    def render(self, mode="human"):
        if not self.renderer:
            from tarware.rendering import Viewer
            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        # Also seed gymnasium's lazily-created `np_random`: it drives outbound
        # request replenishment (`_replenish_outbound`), and without this the
        # request stream is re-seeded from OS entropy in every process, making
        # episodes irreproducible even with a fixed reset seed.
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
