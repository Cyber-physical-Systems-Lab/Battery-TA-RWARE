"""PettingZoo Parallel API with primitive (low-level) actions and per-robot rewards."""
import os

import numpy as np

from battery_tarware import PRIMITIVE_ACTIONS
from battery_tarware.pettingzoo_env import BatteryTarwareParallelEnv

STEPS = int(os.environ.get("EXAMPLE_STEPS", 300))

env = BatteryTarwareParallelEnv("basic", action_mode="primitive", reward_mode="individual", max_steps=STEPS)
obs, infos = env.reset(seed=0)
print("agents:", env.agents)
print("primitive actions:", PRIMITIVE_ACTIONS)
rng = np.random.default_rng(0)
returns = {a: 0.0 for a in env.agents}
while env.agents:
    actions = {a: int(rng.choice(np.flatnonzero(infos[a]["action_mask"]))) for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    for a, r in rewards.items():
        returns[a] += r
print("per-robot returns:", {a: round(r, 2) for a, r in returns.items()})
print("global state vector for centralized critics:", env.state().shape)
