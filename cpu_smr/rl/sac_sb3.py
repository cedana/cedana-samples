#!/usr/bin/env python3

import os
import signal
import sys

import gymnasium as gym
from stable_baselines3 import SAC


def handle_exit(signum, frame):
    sys.exit(1)


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

os.environ['CUDA_VISIBLE_DEVICES'] = ''

env = gym.make('BipedalWalker-v3')

model = SAC(
    'MlpPolicy',
    env,
    verbose=1,
    buffer_size=1_000_000,  # Store the last 1 million transitions
    train_freq=(1, 'episode'),  # Train at the end of each episode
)

print('--- Starting training with SAC  ---')
model.learn(total_timesteps=2_000_000)

env.close()
