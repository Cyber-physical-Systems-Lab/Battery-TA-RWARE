import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from tarware.warehouse import Warehouse, RewardType

class WarehouseMultiAgentEnv(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        
        # Parse env_id to extract parameters
        env_id = env_config["env_id"]
        if "extralarge" in env_id:
            shelf_rows, shelf_columns = 4, 7
            request_queue_size = 60
            num_agvs = 14
            num_pickers = 7
        elif "tiny" in env_id:
            shelf_rows, shelf_columns = 1, 3
            request_queue_size = 20
            num_agvs = 2
            num_pickers = 1
        else:
            # Default fallback
            shelf_rows, shelf_columns = 1, 3
            request_queue_size = 20
            num_agvs = 2
            num_pickers = 1
            
        # Extract observation type
        if "partialobs" in env_id:
            observation_type = "partial"
        elif "globalobs" in env_id:
            observation_type = "global"
        else:
            observation_type = "partial"
            
        # Directly instantiate Warehouse instead of using gym.make
        self.env = Warehouse(
            shelf_columns=shelf_columns,
            column_height=8,
            shelf_rows=shelf_rows,
            num_agvs=num_agvs,
            num_pickers=num_pickers,
            request_queue_size=request_queue_size,
            max_inactivity_steps=None,
            max_steps=500,
            reward_type=RewardType.INDIVIDUAL,
            normalised_coordinates=False,
            observation_type=observation_type,
        )
        
        self._num_agents = num_agvs + num_pickers
        # For TA-RWARE, observation_space is a Tuple of spaces for each agent
        if isinstance(self.env.observation_space, gym.spaces.Tuple):
            self._observation_space = gym.spaces.Dict({f"agent_{i}": self.env.observation_space[i] for i in range(self._num_agents)})
        else:
            self._observation_space = self.env.observation_space
        
        # For TA-RWARE, action_space is a Tuple of spaces for each agent  
        if isinstance(self.env.action_space, gym.spaces.Tuple):
            self._action_space = gym.spaces.Dict({f"agent_{i}": self.env.action_space[i] for i in range(self._num_agents)})
        else:
            self._action_space = self.env.action_space
        
        # Set agents and possible_agents for Ray MultiAgentEnv
        self.agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.possible_agents = self.agents.copy()

    @property
    def num_agents(self):
        return self._num_agents

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, *, seed=None, options=None):
        obs = self.env.reset(seed=seed, options=options)
        return {f"agent_{i}": obs[i] for i in range(self.num_agents)}, {}

    def step(self, action_dict):
        actions = [action_dict[f"agent_{i}"] for i in range(self.num_agents)]
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        done = {"__all__": all(terminated) or all(truncated)}
        obs_dict = {f"agent_{i}": obs[i] for i in range(self.num_agents)}
        rew_dict = {f"agent_{i}": rewards[i] for i in range(self.num_agents)}
        done.update({f"agent_{i}": terminated[i] or truncated[i] for i in range(self.num_agents)})
        # Convert info to MultiAgentDict format
        info_dict = {"__common__": info}
        return obs_dict, rew_dict, done, done, info_dict

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()