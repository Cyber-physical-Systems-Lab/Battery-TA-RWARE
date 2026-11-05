"""Gymnasium environment.

* observation: ``Dict({robot_name: Dict(self, robots, tasks, chargers, action_mask)})``
* action: ``MultiDiscrete([n_actions] * n_robots)`` (one action per robot, in fleet order)
* reward: team reward (sum over robots); per-robot rewards in ``info["robot_rewards"]``
* terminated: demand finished (finite demand) or every robot stranded; truncated: ``max_steps``

Macro actions (``action_mode="macro"``), per robot::

    0                    no-op: keep doing what it is doing
    1 .. K               join the task in board slot k (first open role it can fill)
    K+1 .. K+C           go to charger c and charge until full
    K+C+1                release its current commitment

Primitive actions (``action_mode="primitive"``)::

    0 wait, 1 up, 2 down, 3 left, 4 right,
    5 interact (charge on a dock, or join a task it is positioned for), 6 release
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from battery_tarware.config import ScenarioConfig
from battery_tarware.observation import ObservationBuilder
from battery_tarware.rendering import HumanViewer, ansi, rgb_array
from battery_tarware.simulator import Simulator


class BatteryTarwareEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "ansi", "human"], "render_fps": 8}

    def __init__(self, scenario: Union[str, dict, ScenarioConfig] = "basic", render_mode: Optional[str] = None,
                 **overrides: Any):
        from battery_tarware.scenarios import resolve_scenario
        self.config = resolve_scenario(scenario, **overrides)
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError("render_mode must be one of %s" % self.metadata["render_modes"])
        self.render_mode = render_mode
        self.sim = Simulator(self.config)
        self.obs_builder = ObservationBuilder(self.sim)
        self.robot_names = [r.name for r in self.sim.robots]
        per_robot = self.obs_builder.robot_space()
        self.observation_space = spaces.Dict({n: per_robot for n in self.robot_names})
        self.action_space = spaces.MultiDiscrete([self.obs_builder.n_actions] * len(self.robot_names))
        self._viewer = None

    # ---- gymnasium API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(2 ** 31 - 1))
        self.sim.reset(seed)
        obs = self.obs_builder.build(self.sim.action_masks())
        if self.render_mode == "human":
            self.render()
        return obs, self._info([], 0, np.zeros(len(self.robot_names)))

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        out = self.sim.step(action)
        masks = self.sim.action_masks()
        obs = self.obs_builder.build(masks)
        if self.render_mode == "human":
            self.render()
        return obs, out.reward, out.terminated, out.truncated, self._info(out.events, out.invalid_actions, out.robot_rewards)

    def _info(self, events, invalid, robot_rewards) -> Dict[str, Any]:
        return {"robot_rewards": np.asarray(robot_rewards, dtype=np.float64), "events": events,
                "invalid_actions": int(invalid), "metrics": self.sim.metrics(),
                "action_mask": self.sim.action_masks()}

    def render(self):
        if self.render_mode == "ansi":
            return ansi(self.sim)
        img = rgb_array(self.sim)
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = HumanViewer()
            self._viewer.show(img, "%s  tick %d" % (self.config.name, self.sim.tick))
            return None
        return img

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ---- research API
    def state_view(self):
        from battery_tarware.state_view import build_state_view
        return build_state_view(self.sim)

    def action_masks(self) -> np.ndarray:
        """(n_robots, n_actions) int8, for masked policies."""
        return self.sim.action_masks()
