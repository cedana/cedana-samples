import os
import signal
import sys

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations


def handle_exit(signum, frame):
    sys.exit(1)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Configure the algorithm.
config = (
    PPOConfig()
    .environment('Taxi-v3')
    .env_runners(
        num_env_runners=2,
        # Observations are discrete (ints) -> We need to flatten (one-hot) them.
        env_to_module_connector=lambda env: FlattenObservations(),
    )
    .evaluation(evaluation_num_env_runners=1)
)

from pprint import pprint

# Build the algorithm.
algo = config.build_algo()

# Train it for 25 iterations ...
for _ in range(25):
    pprint(algo.train())


# ... and evaluate it.
pprint(algo.evaluate())

# Release the algo's resources (remote actors, like EnvRunners and Learners).
algo.stop()
