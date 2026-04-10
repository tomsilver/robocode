"""Test the full approach on multiple seeds."""
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

def load_primitives():
    import importlib.util
    spec = importlib.util.spec_from_file_location("motion_planning", "primitives/motion_planning.py")
    mp = importlib.util.load_from_spec(spec) if hasattr(importlib.util, 'load_from_spec') else None
    # Use the simpler approach
    import sys
    sys.path.insert(0, 'primitives')
    from motion_planning import BiRRT
    return {'BiRRT': BiRRT}

def run_episode(seed, max_steps=500, verbose=True):
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=seed)

    primitives = load_primitives()

    from approach import GeneratedApproach
    approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
    approach.reset(obs, info)

    total_reward = 0
    for step in range(max_steps):
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            if verbose:
                print(f"Seed {seed}: Done at step {step+1}, total_reward={total_reward}")
            return True, step+1, total_reward
    if verbose:
        print(f"Seed {seed}: Not done after {max_steps} steps, total_reward={total_reward}")
    return False, max_steps, total_reward

if __name__ == '__main__':
    seeds = [0, 1, 2, 3, 42]
    results = []
    for seed in seeds:
        success, steps, reward = run_episode(seed, max_steps=600)
        results.append((seed, success, steps, reward))

    print("\n=== Summary ===")
    n_success = sum(1 for _, s, _, _ in results if s)
    print(f"Success: {n_success}/{len(results)}")
    for seed, success, steps, reward in results:
        print(f"  seed={seed}: {'OK' if success else 'FAIL'} steps={steps} reward={reward}")
