import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, is_holding, NUM_OBSTRUCTIONS, is_obstruction_blocking_surface

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}

obs, info = env.reset(seed=42)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

done = False
steps = 0

while not done and steps < 200:
    cur = approach._current
    phase = getattr(cur, '_phase', '?')
    r = extract_robot(obs)
    if phase == 'NAV_DOWN' and steps > 46:
        if steps % 10 == 0:
            print(f"step={steps} NAV_DOWN robot=({r['x']:.3f},{r['y']:.3f}) path_step={cur._path_step}/{len(cur._path)}")
            if cur._path_step < len(cur._path):
                wp = cur._path[cur._path_step]
                print(f"  waypoint=({wp[0]:.3f},{wp[1]:.3f}) pick_y={cur._pick_y:.3f}")

    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    steps += 1
