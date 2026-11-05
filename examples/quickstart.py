"""Gymnasium quickstart: a masked random policy and the greedy coalition baseline."""
import os

import gymnasium as gym
import numpy as np

import battery_tarware as bt  # registers the environments

STEPS = int(os.environ.get("EXAMPLE_STEPS", 500))

env = gym.make("battery_tarware/Intermediate-v2", max_steps=STEPS)
obs, info = env.reset(seed=0)
print("robots:", list(obs))
print("action space:", env.action_space)

rng = np.random.default_rng(0)
total = 0.0
for _ in range(STEPS):
    masks = info["action_mask"]                                 # (n_robots, n_actions)
    action = [rng.choice(np.flatnonzero(m)) for m in masks]     # random *valid* action per robot
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    if terminated or truncated:
        break
print("random policy:", info["metrics"]["tasks_completed"], "tasks, return %.1f" % total)

metrics = bt.run_episode(bt.make("intermediate", max_steps=STEPS), bt.GreedyCoalitionCoordinator("edd"), seed=0)
print("greedy coalition baseline (EDD):", metrics["tasks_completed"], "tasks, return %.1f" % metrics["return"])
