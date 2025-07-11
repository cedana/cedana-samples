import os
import signal
import sys

import gymnasium as gym
from stable_baselines3 import PPO


def handle_exit(signum, frame):
    sys.exit(1)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

os.environ['CUDA_VISIBLE_DEVICES'] = ''

env = gym.make('Taxi-v3')
model = PPO('MlpPolicy', env, verbose=1)

print('--- Starting long-running training with Stable Baselines3 ---')
model.learn(total_timesteps=1_000_000)

print('\n--- Training complete ---')

env.close()
