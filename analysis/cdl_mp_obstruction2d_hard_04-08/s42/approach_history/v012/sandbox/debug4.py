import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, NUM_OBSTRUCTIONS

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}

obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

done = False
steps = 0
in_second_hzdrop = False
second_start = 0

while not done and steps < 200:
    cur = approach._current
    phase = getattr(cur, '_phase', '?')
    
    # Detect second NAV_HZDROP
    if phase == 'NAV_HZDROP' and steps > 50 and not in_second_hzdrop:
        in_second_hzdrop = True
        second_start = steps
        r = extract_robot(obs)
        print(f"Second NAV_HZDROP at step={steps}")
        print(f"  Robot: ({r['x']:.3f},{r['y']:.3f}) theta={r['theta']:.2f} arm_joint={r['arm_joint']:.3f}")
        print(f"  drop_x={cur._drop_x:.3f}, drop_zone_idx={cur._drop_zone_idx}")
        print(f"  path len={len(cur._path)}, path_step={cur._path_step}")
        if cur._path:
            print(f"  path[-1]={cur._path[-1]}")
        # Print all obstruction positions
        for i in range(NUM_OBSTRUCTIONS):
            ob = extract_obstruction(obs, i)
            print(f"  Obs{i}: ({ob['x']:.3f},{ob['y']:.3f}) w={ob['width']:.3f} h={ob['height']:.3f}")

    if in_second_hzdrop and steps % 20 == 0:
        r = extract_robot(obs)
        print(f"  step={steps} robot=({r['x']:.3f},{r['y']:.3f}) path_step={cur._path_step}/{len(cur._path)}")
        if cur._path_step < len(cur._path):
            wp = cur._path[cur._path_step]
            print(f"    -> waypoint=({wp[0]:.3f},{wp[1]:.3f})")

    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    steps += 1
