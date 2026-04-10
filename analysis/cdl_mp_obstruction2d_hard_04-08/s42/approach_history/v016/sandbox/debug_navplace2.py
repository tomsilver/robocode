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

in_nav = False
count = 0
for step in range(600):
    cur = approach._current
    bname = type(cur).__name__
    ph = getattr(cur, '_phase', '?')
    if bname == 'PlaceTargetBlock' and ph == 'NAV_PLACE':
        if not in_nav:
            in_nav = True
        count += 1
        if count <= 5 or count % 50 == 0:
            r = extract_robot(obs)
            pl = cur
            print(f"  step={step} count={count}: robot=({r['x']:.4f},{r['y']:.4f}), path={[f'({p[0]:.4f},{p[1]:.4f})' for p in getattr(pl,'_path',[])]}, ps={getattr(pl,'_path_step','?')}")
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Done at step {step+1}")
        break
print(f"Nav steps: {count}")
