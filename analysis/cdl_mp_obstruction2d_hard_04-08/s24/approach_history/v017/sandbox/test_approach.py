"""Test the full approach on multiple seeds."""
import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach

primitives = {"BiRRT": BiRRT}

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")

for seed in [0, 1, 2, 3, 42]:
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
    approach.reset(obs, info)

    done = False
    total_reward = 0
    steps = 0
    max_steps = 1000

    while not done and steps < max_steps:
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"Seed {seed}: steps={steps}, reward={total_reward:.1f}, terminated={terminated}, truncated={truncated}")

env.close()
