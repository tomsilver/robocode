import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}
for seed in [0, 1, 2, 42, 7, 13, 99, 100]:
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
    approach.reset(obs, info)
    done = False; steps = 0; total_reward = 0.0
    while not done and steps < 2000:
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated; total_reward += reward; steps += 1
    print(f"Seed {seed:3d}: steps={steps:4d}, reward={total_reward:6.0f}, terminated={terminated}")
