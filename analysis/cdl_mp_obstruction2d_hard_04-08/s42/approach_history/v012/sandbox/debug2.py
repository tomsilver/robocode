import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}

obs, info = env.reset(seed=42)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

done = False
steps = 0
printed = False

while not done and steps < 200:
    cur = approach._current
    phase = getattr(cur, '_phase', '?')
    if phase == 'NAV_DROP_DESCEND' and not printed:
        r = extract_robot(obs)
        print(f"Robot: x={r['x']:.3f} y={r['y']:.3f}")
        print(f"pick_y={cur._pick_y:.3f} drop_x={cur._drop_x:.3f}")
        print(f"path len={len(cur._path)}")
        if cur._path:
            print(f"path[0]={cur._path[0]} last={cur._path[-1]}")
        print(f"base_radius={r['base_radius']:.3f}")
        from act_helpers import TABLE_HEIGHT
        print(f"TABLE_HEIGHT={TABLE_HEIGHT}")
        min_y = TABLE_HEIGHT + r['base_radius']
        print(f"min_y (TABLE_HEIGHT + base_radius) = {min_y:.3f}")
        printed = True

    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    steps += 1
