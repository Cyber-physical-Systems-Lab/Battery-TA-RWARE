import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from train_utils import WarehouseMultiAgentEnv
from custom_model import WarehouseMARLModel  # Import custom model
from ray.rllib.models import ModelCatalog  # Import ModelCatalog

# Register the custom model at module level so all worker processes can access it
ModelCatalog.register_custom_model("warehouse_marl_model", WarehouseMARLModel)

def train():
    ray.init(ignore_reinit_error=True)

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)  # Use old API stack for custom models
        .environment(
            env=WarehouseMultiAgentEnv,
            env_config={"env_id": "tarware-extralarge-14agvs-7pickers-partialobs-chg-v1"},
        )
        .multi_agent(
            policies={
                f"agent_{i}": (
                    None,  # Use default policy class
                    None,  # observation space (auto-inferred)
                    None,  # action space (auto-inferred)
                    {
                        "model": {
                            "custom_model": "warehouse_marl_model",  # Use registered model name
                        }
                    }
                ) for i in range(21)
            },
            policy_mapping_fn=lambda agent_id, episode, worker=None, **kwargs: agent_id,
        )
        .resources(
            num_gpus=1,
            num_cpus_per_worker=1,
        )
        .env_runners(
            num_env_runners=3,
            num_gpus_per_env_runner=0,  # No GPU for env runners
        )
        .learners(
            num_gpus_per_learner=1,  # GPU for learner
        )
        .training(
            lr=5e-5,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            num_epochs=10,
            minibatch_size=256,
            train_batch_size=4000,
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=5,
            evaluation_config={"explore": False},
        )
        .framework('torch')
    )

    # Enable rendering
    config.render_env = True

    # Run training directly instead of using tune.run
    algo = config.build()

    print("✓ Algorithm built successfully with WarehouseMARLModel!")

    # Run full training for 100 iterations
    for i in range(100):
        result = algo.train()
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}: episode_return_mean = {result['env_runners']['episode_return_mean']:.3f}")

    algo.stop()
    print("✓ Full training completed successfully with custom WarehouseMARLModel!")

if __name__ == "__main__":
    train()