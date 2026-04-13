"""Test the full approach on multiple seeds."""
import sys
sys.path.insert(0, '/sandbox')

import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv("kinder/ClutteredStorage2D-b3-v0")

seeds = [0, 1, 2, 42]
results = []

for seed in seeds:
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
    approach.reset(obs, info)

    total_reward = 0
    done = False
    step = 0
    max_steps = 2000

    while not done and step < max_steps:
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step += 1

    results.append({
        'seed': seed, 'steps': step, 'reward': total_reward,
        'terminated': terminated, 'truncated': truncated,
    })
    print(f"Seed {seed}: steps={step}, reward={total_reward:.1f}, terminated={terminated}, truncated={truncated}")

print(f"\nSolved {sum(r['terminated'] for r in results)}/{len(seeds)} episodes")
