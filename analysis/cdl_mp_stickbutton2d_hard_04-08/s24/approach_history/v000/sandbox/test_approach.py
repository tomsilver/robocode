"""Test the full approach on multiple seeds."""
import numpy as np
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/primitives')

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach

env = KinderGeom2DEnv("kinder/StickButton2D-b5-v0")
primitives = {"BiRRT": BiRRT}

results = []
for seed in [0, 1, 2, 42]:
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
    approach.reset(obs, info)

    total_reward = 0
    steps = 0
    done = False
    max_steps = 3000

    while not done and steps < max_steps:
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    from obs_helpers import get_unpressed_buttons
    unpressed = get_unpressed_buttons(obs)
    status = "SOLVED" if terminated else f"FAILED ({len(unpressed)} unpressed)"
    print(f"Seed {seed:3d}: {steps:4d} steps | reward {total_reward:6.1f} | {status}")
    results.append(terminated)

print(f"\nSolved: {sum(results)}/{len(results)}")
