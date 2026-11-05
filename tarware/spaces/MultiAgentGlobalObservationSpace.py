"""
Global observation builder for Tarware agents.

Each agent receives a fixed-width vector containing its own state first,
followed by the other agents, shelf request/inbound bits for all rack
locations, and normalized battery levels for every agent. AGV entries include
carrying/outbound and pending-load-action state; coordinates can be absolute or
normalized by the environment.
"""

import numpy as np
from gymnasium import spaces

from tarware.definitions import Action, AgentType, CollisionLayers
from tarware.spaces.MultiAgentBaseObservationSpace import (
    MultiAgentBaseObservationSpace, _VectorWriter)


class MultiAgentGlobalObservationSpace(MultiAgentBaseObservationSpace):
    def __init__(self, num_agvs, num_pickers, grid_size, shelf_locations, normalised_coordinates=False):
        super(MultiAgentGlobalObservationSpace, self).__init__(num_agvs, num_pickers, grid_size, shelf_locations, normalised_coordinates)

        self._define_obs_length()
        self.obs_lengths = [self.obs_length for _ in range(self.num_agents)]
        self._current_agents_info = []
        self._current_shelves_info = []
        self._current_batteries = []

        ma_spaces = []
        for obs_length in self.obs_lengths:
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(obs_length,),
                    dtype=np.float32,
                )
            ]

        self.ma_spaces = spaces.Tuple(tuple(ma_spaces))

    def _define_obs_length(self):
        location_space = spaces.Box(low=0.0, high=max(self.grid_size), shape=(2,), dtype=np.float32)

        self.obs_bits_for_agvs = (3 + spaces.flatdim(location_space)  + spaces.flatdim(location_space)) * self.num_agvs
        self.obs_bits_for_pickers = (spaces.flatdim(location_space)  + spaces.flatdim(location_space)) * self.num_pickers
        self.obs_bits_per_shelf = 1 * self.shelf_locations
        self.obs_bits_for_requests = 1 * self.shelf_locations
        # Battery level (normalized 0-1) for each agent — enables energy-aware policies
        self.obs_bits_for_batteries = self.num_agents
        self.obs_length = (
            self.obs_bits_for_agvs
            + self.obs_bits_for_pickers
            + self.obs_bits_per_shelf
            + self.obs_bits_for_requests
            + self.obs_bits_for_batteries
        )

    def extract_environment_info(self, environment):
        self._current_agents_info = []
        self._current_shelves_info = []
        self._current_batteries = []

        # Extract battery levels (normalized to [0, 1])
        for agent in environment.agents:
            self._current_batteries.append(agent.battery / 100.0)

        # Extract agents info
        for agent in environment.agents:
            agent_info = []
            if agent.type in (AgentType.AGV, AgentType.AGENT):
                if agent.carrying_shelf:
                    # bit 1: carrying a shelf; bit 2: 1=outbound, 0=inbound
                    is_outbound = int(agent.carrying_shelf in environment.request_queue
                                      or agent.carrying_shelf.direction.name == "OUT")
                    agent_info.extend([1, is_outbound])
                else:
                    agent_info.extend([0, 0])
                agent_info.extend([agent.req_action == Action.TOGGLE_LOAD])
            agent_info.extend(self.process_coordinates((agent.y, agent.x), environment))
            if agent.target:
                agent_info.extend(self.process_coordinates(environment.action_id_to_coords_map[agent.target], environment))
            else:
                agent_info.extend([0, 0])
            self._current_agents_info.append(agent_info)

        # Extract shelves info
        for group in environment.rack_groups:
            for (x, y) in group:
                id_shelf = environment.grid[CollisionLayers.SHELVES, x, y]
                if id_shelf!=0:
                    shelf = environment.shelfs[id_shelf - 1]
                    # bit 1: outbound requested; bit 2: inbound (shelf away, goal has it)
                    self._current_shelves_info.extend([
                        int(shelf in environment.request_queue),
                        int(shelf in environment.inbound_queue),
                    ])
                else:
                    self._current_shelves_info.extend([0, 0])

    def observation(self, agent):
        obs = _VectorWriter(self.ma_spaces[agent.id - 1].shape[0])
        obs.write(self._current_agents_info[agent.id - 1])
        for agent_id, agent_info in enumerate(self._current_agents_info):
            if agent_id != agent.id - 1:
                obs.write(agent_info)
        obs.write(self._current_shelves_info)
        # Battery levels for all agents (energy coupling awareness)
        obs.write(self._current_batteries)
        return obs.vector
