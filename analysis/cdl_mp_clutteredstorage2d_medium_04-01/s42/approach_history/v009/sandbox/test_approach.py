"""Test the full approach on multiple seeds."""
import sys
sys.path.insert(0, '/sandbox')

import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT

from approach import GeneratedApproach
from obs_helpers import get_outside_blocks, is_block_in_shelf, BLOCK_NAMES

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')

for seed in [0, 1, 2, 3, 42]:
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
    approach.reset(obs, info)

    outside_init = get_outside_blocks(obs)
    print(f"Seed {seed}: outside blocks initially = {outside_init}")

    done = False
    steps = 0
    max_steps = 2000
    total_reward = 0.0

    while not done and steps < max_steps:
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    in_shelf = [bn for bn in BLOCK_NAMES if is_block_in_shelf(obs, bn)]
    print(f"  steps={steps}, total_reward={total_reward:.1f}, terminated={terminated}, in_shelf={in_shelf}")
    print()

print("Done!")
