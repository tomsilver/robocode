"""Test the full approach on multiple seeds."""
import sys
sys.path.insert(0, '/sandbox')

import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv

def test_seed(seed, max_steps=800):
    from approach import GeneratedApproach
    env = KinderGeom2DEnv("kinder/ClutteredStorage2D-b3-v0")
    obs, info = env.reset(seed=seed)

    # Build primitives
    import sys
    sys.path.insert(0, '/sandbox')
    from primitives.motion_planning import BiRRT
    primitives = {'BiRRT': BiRRT}

    approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
    approach.reset(obs, info)

    total_reward = 0
    for step in range(max_steps):
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print(f"Seed {seed}: SOLVED in {step+1} steps, reward={total_reward:.1f}")
            return True
        if truncated:
            break

    print(f"Seed {seed}: FAILED after {max_steps} steps, reward={total_reward:.1f}")
    return False

if __name__ == "__main__":
    results = []
    for seed in [0, 1, 2, 3, 42]:
        try:
            r = test_seed(seed)
            results.append(r)
        except Exception as e:
            import traceback
            print(f"Seed {seed}: ERROR - {e}")
            traceback.print_exc()
            results.append(False)
    print(f"\nResults: {sum(results)}/{len(results)} solved")
