import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}
obs, info = env.reset(seed=1)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

count = 0
for step in range(270):
    cur = approach._current
    bname = type(cur).__name__
    ph = getattr(cur, '_phase', '?')
    if bname == 'PlaceTargetBlock' and ph == 'NAV_PLACE':
        count += 1
        r = extract_robot(obs)
        action = approach.get_action(obs)
        if 4 <= count <= 12 or (35 <= count <= 45):
            print(f"  count={count}: pos=({r['x']:.4f},{r['y']:.4f}), action_dy={action[1]:.4f}")
        obs, reward, terminated, truncated, info = env.step(action)
    else:
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("Done!")
        break
