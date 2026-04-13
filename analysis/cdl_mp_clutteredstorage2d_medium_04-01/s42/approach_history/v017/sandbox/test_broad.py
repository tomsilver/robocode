import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import is_block_in_shelf, BLOCK_NAMES

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
seeds = list(range(20)) + [42, 100, 200, 999]
passed = failed = 0
for seed in seeds:
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
    approach.reset(obs, info)
    done = False; steps = 0
    while not done and steps < 2000:
        obs, _, terminated, truncated, _ = env.step(approach.get_action(obs))
        done = terminated or truncated; steps += 1
    in_shelf = [bn for bn in BLOCK_NAMES if is_block_in_shelf(obs, bn)]
    ok = terminated and len(in_shelf) == 3
    print(f"Seed {seed:3d}: {'OK' if ok else 'FAIL'} steps={steps} in_shelf={in_shelf}")
    if ok: passed += 1
    else: failed += 1
print(f"\nPassed: {passed}/{passed+failed}")
