"""Test approach on multiple seeds."""
import sys
import os
sys.path.insert(0, '/sandbox')
os.chdir('/sandbox')

import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach
from obs_helpers import get_blocks_outside_shelf, is_block_in_shelf, BLOCK_NAMES
from primitives.motion_planning import BiRRT

PRIMITIVES = {'BiRRT': BiRRT}


def run_episode(seed, max_steps=2000):
    env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
    obs, info = env.reset(seed=seed)

    approach = GeneratedApproach(env.action_space, env.observation_space, PRIMITIVES)
    approach.reset(obs, info)

    total_reward = 0
    for step in range(max_steps):
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print(f"  seed={seed}: SUCCESS at step {step+1}, reward={total_reward:.0f}")
            return True, step+1, total_reward
        if truncated:
            break

    outside = get_blocks_outside_shelf(obs)
    print(f"  seed={seed}: FAILED after {max_steps} steps, reward={total_reward:.0f}, outside={outside}")
    return False, max_steps, total_reward


print("Testing approach...")
results = []
for seed in [0, 1, 2, 42]:
    r = run_episode(seed, max_steps=2000)
    results.append(r)

successes = sum(1 for r in results if r[0])
print(f"\nTotal: {successes}/{len(results)} succeeded")
