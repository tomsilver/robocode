import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
results = {}
for seed in range(30):
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, {})
    approach.reset(obs, info)
    total_reward = 0
    done = False
    steps = 0
    while not done and steps < 2000:
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    results[seed] = terminated
    if not terminated:
        print(f"FAIL seed={seed}")

successes = sum(results.values())
print(f"\nSuccess rate: {successes}/30")
