import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, NUM_OBSTRUCTIONS

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}
obs, info = env.reset(seed=1)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

in_nav = False
count = 0
for step in range(310):
    cur = approach._current
    bname = type(cur).__name__
    ph = getattr(cur, '_phase', '?')
    if bname == 'PlaceTargetBlock' and ph == 'NAV_PLACE':
        count += 1
        r = extract_robot(obs)
        if 40 <= count <= 55:
            print(f"  count={count} step={step}: robot=({r['x']:.4f},{r['y']:.4f}) vac={r['vacuum']:.0f}")
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("Done!")
        break
