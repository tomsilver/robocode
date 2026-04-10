import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import *
from act_helpers import *

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}

seed = 42
obs, info = env.reset(seed=seed)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

max_steps = 500
prev_robot_x = None
stuck_count = 0
for step in range(max_steps):
    b = approach._current
    phase = b._phase if hasattr(b, '_phase') else '?'
    robot = extract_robot(obs)
    
    if robot['x'] == prev_robot_x and step > 50:
        stuck_count += 1
        if stuck_count == 5:  # just stuck for 5 steps
            print(f"\nSTUCK at step {step}, phase={phase}")
            print(f"Robot: ({robot['x']:.3f},{robot['y']:.3f})")
            for i in range(4):
                ob = extract_obstruction(obs, i)
                print(f"Obs[{i}]: bl=({ob['x']:.3f},{ob['y']:.3f}), w={ob['width']:.3f}, h={ob['height']:.3f}")
            block = extract_target_block(obs)
            print(f"Block: bl=({block['x']:.3f},{block['y']:.3f})")
            surf = extract_target_surface(obs)
            print(f"Surf: bl=({surf['x']:.3f},{surf['y']:.3f}), w={surf['width']:.3f}")
            print(f"surf_center_x={surf['x']+surf['width']/2:.3f}")
            if hasattr(b, '_path') and b._path:
                print(f"Path target: {b._path[-1]}")
            # Try applying left action manually
            print("Testing leftward move...")
            break
    else:
        stuck_count = 0
    prev_robot_x = robot['x']
    
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"DONE at step {step}!")
        break
