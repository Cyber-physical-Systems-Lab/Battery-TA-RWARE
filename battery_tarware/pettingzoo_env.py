"""PettingZoo ParallelEnv wrapper (``pip install 'battery-tarware[pettingzoo]'``).

Agents are robot names. Each agent observes its own Dict observation and gets its own reward
(``reward_mode="individual"``, credit as in ``info["robot_rewards"]``) or the team reward
(``reward_mode="team"``). Stranded robots stay in the episode (their actions are masked to no-op),
so the agent set is fixed for the whole episode.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    from pettingzoo import ParallelEnv
except ImportError as e:  # pragma: no cover
    raise ImportError("battery_tarware.pettingzoo_env needs pettingzoo: pip install 'battery-tarware[pettingzoo]'") from e

from gymnasium import spaces

from battery_tarware.env import BatteryTarwareEnv


class BatteryTarwareParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["rgb_array", "ansi", "human"], "name": "battery_tarware_v1"}

    def __init__(self, scenario="basic", render_mode: Optional[str] = None, reward_mode: str = "individual",
                 **overrides: Any):
        if reward_mode not in ("individual", "team"):
            raise ValueError("reward_mode must be individual|team")
        self.env = BatteryTarwareEnv(scenario, render_mode=render_mode, **overrides)
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.possible_agents = list(self.env.robot_names)
        self.agents = list(self.possible_agents)
        self._obs_space = self.env.observation_space[self.possible_agents[0]]
        self._act_space = spaces.Discrete(self.env.obs_builder.n_actions)

    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._act_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        return obs, {a: {"action_mask": obs[a]["action_mask"]} for a in self.agents}

    def step(self, actions: Dict[str, int]):
        joint = np.array([int(actions.get(a, 0)) for a in self.possible_agents])
        obs, reward, term, trunc, info = self.env.step(joint)
        rr = info["robot_rewards"]
        rewards = {a: float(reward if self.reward_mode == "team" else rr[i]) for i, a in enumerate(self.possible_agents)}
        terms = {a: bool(term) for a in self.agents}
        truncs = {a: bool(trunc) for a in self.agents}
        infos = {a: {"action_mask": obs[a]["action_mask"]} for a in self.agents}
        if term or trunc:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def state(self) -> np.ndarray:
        """Global state for centralized critics: robot cells, SoC, status and task of every robot."""
        a = self.env.state_view().to_arrays()
        H, W = self.env.sim.layout.shape
        cells = a["robot_cell"].astype(np.float32) / np.array([max(1, H - 1), max(1, W - 1)], np.float32)
        return np.concatenate([cells.ravel(), a["robot_soc"].astype(np.float32),
                               a["robot_status"].astype(np.float32), (a["robot_task"] >= 0).astype(np.float32)])
