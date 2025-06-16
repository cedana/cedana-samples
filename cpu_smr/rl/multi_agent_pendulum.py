import os
import gymnasium as gym

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.examples.envs.classes.multi_agent import MultiAgentPendulum
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env

# --- Create a parser with long-running defaults ---
parser = add_rllib_example_script_args(
    # Set a high timestep count to ensure a long run.
    default_timesteps=3000000,
    # Set iterations to a very high number (effectively infinite).
    default_iters=100000000,
    default_reward=-400.0,
)
parser.add_argument("--num-policies", type=int, default=2)


if __name__ == "__main__":
    # This line forces the script to run on the CPU, even if GPUs are available.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    args = parser.parse_args()

    # Register the custom environment with Tune.
    # Note: Increasing --num-agents on the command line will also make the
    # problem harder and the run longer.
    # e.g. python [script name].py --num-agents=10
    if args.num_agents > 0:
        register_env(
            "env",
            lambda config: MultiAgentPendulum(config={"num_agents": args.num_agents}),
        )

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        # Use our custom environment if --num-agents > 0, otherwise use single-agent.
        .environment("env" if args.num_agents > 0 else "Pendulum-v1")
        .training(
            # Standard hyperparameters for PPO on continuous control tasks.
            train_batch_size_per_learner=512,
            minibatch_size=64,
            lambda_=0.1,
            gamma=0.95,
            lr=0.0003,
            vf_clip_param=10.0,
        )
        .rl_module(
            # Model configuration now correctly consolidated here.
            model_config=DefaultModelConfig(
                fcnet_hiddens=[256, 256],
                fcnet_activation="relu"
            ),
        )
    )

    # Add the multi-agent configuration if specified.
    if args.num_agents > 0:
        base_config.multi_agent(
            # Create a policy for each agent.
            policies={f"p{i}" for i in range(args.num_agents)},
            # Map agent ID to its policy ID (e.g., agent "0" uses policy "p0").
            policy_mapping_fn=lambda agent_id, *a, **kw: f"p{agent_id}",
        )

    # Run the experiment.
    run_rllib_example_script_experiment(base_config, args)
