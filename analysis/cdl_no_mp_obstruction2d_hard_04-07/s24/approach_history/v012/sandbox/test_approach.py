"""Test the full approach on multiple seeds."""
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')

results = {}
for seed in [0, 1, 2, 42]:
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, {})
    approach.reset(obs, info)

    total_reward = 0
    done = False
    steps = 0
    max_steps = 2000
    while not done and steps < max_steps:
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    results[seed] = {
        'steps': steps,
        'reward': total_reward,
        'success': terminated,
    }
    print(f"seed={seed}: steps={steps}, reward={total_reward:.0f}, success={terminated}")

successes = sum(1 for r in results.values() if r['success'])
print(f"\nSuccess rate: {successes}/{len(results)}")
