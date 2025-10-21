import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from train_utils import WarehouseMultiAgentEnv
import numpy as np

def evaluate(checkpoint_path, num_episodes=10):
    ray.init(ignore_reinit_error=True)
    
    # Register the custom environment
    from ray import tune
    tune.register_env("WarehouseMultiAgentEnv", lambda config: WarehouseMultiAgentEnv(config))
    
    config = PPOConfig().environment(
        env="WarehouseMultiAgentEnv",
        env_config={"env_id": "tarware-tiny-2agvs-1pickers-globalobs-chg"},
    ).multi_agent(
        policies={f"agent_{i}": (None, None, None, {}) for i in range(3)},
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
    )
    
    agent = PPO(config=config)
    agent.restore(checkpoint_path)
    
    env = WarehouseMultiAgentEnv({"env_id": "tarware-tiny-2agvs-1pickers-globalobs-chg"})
    
    total_deliveries = 0
    total_clashes = 0
    total_stucks = 0
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = {"__all__": False}
        episode_length = 0
        while not done["__all__"]:
            actions = {}
            for agent_id in obs.keys():
                actions[agent_id] = agent.compute_single_action(obs[agent_id], policy_id=agent_id)
            obs, rewards, done, _, info = env.step(actions)
            episode_length += 1
        total_deliveries += info.get("shelf_deliveries", 0)
        total_clashes += info.get("clashes", 0)
        total_stucks += info.get("stucks", 0)
        episode_lengths.append(episode_length)
    
    print(f"Average Deliveries: {total_deliveries / num_episodes}")
    print(f"Average Clashes: {total_clashes / num_episodes}")
    print(f"Average Stucks: {total_stucks / num_episodes}")
    print(f"Average Episode Length: {np.mean(episode_lengths)}")
    print(f"Pick Rate (deliveries/hour, assuming 5 steps/sec): {(total_deliveries / num_episodes) * 3600 / (5 * np.mean(episode_lengths)):.2f}")

if __name__ == "__main__":
    checkpoint_path = "/home/xuezhi/task-assignment-robotic-warehouse/checkpoints/PPO/PPO_WarehouseMultiAgentEnv_xxx/checkpoint_xxx"  # Replace with actual path
    evaluate(checkpoint_path)