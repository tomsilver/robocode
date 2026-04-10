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

for step in range(220):
    cur = approach._current
    bname = type(cur).__name__
    ph = getattr(cur, '_phase', '?')
    
    if bname == 'ClearAllObstructions' and ph == 'RELEASE':
        r = extract_robot(obs)
        tgt = getattr(cur, '_target_idx', '?')
        drop_x = getattr(cur, '_drop_x', '?')
        ob = extract_obstruction(obs, tgt)
        print(f"Step {step}: RELEASE obs[{tgt}]: robot=({r['x']:.4f},{r['y']:.4f}), ob_bl=({ob['x']:.4f},{ob['y']:.4f}), drop_x={drop_x:.4f}")
        # Print all obstructions
        for i in range(NUM_OBSTRUCTIONS):
            o = extract_obstruction(obs, i)
            print(f"  obs[{i}]: ({o['x']:.4f},{o['y']:.4f}) w={o['width']:.4f} h={o['height']:.4f}")
    
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Done at step {step+1}")
        break
