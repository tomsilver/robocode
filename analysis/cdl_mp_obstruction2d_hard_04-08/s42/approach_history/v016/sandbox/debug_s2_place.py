import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_target_block, extract_target_surface

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}
obs, info = env.reset(seed=2)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

for step in range(200):
    cur = approach._current
    bname = type(cur).__name__
    ph = getattr(cur, '_phase', '?')
    if bname == 'PlaceTargetBlock' and ph in ('PLAN_PLACE', 'NAV_PLACE'):
        r = extract_robot(obs)
        b = extract_target_block(obs)
        s = extract_target_surface(obs)
        surf_x = getattr(cur, '_surf_x', '?')
        place_y = getattr(cur, '_place_y', '?')
        path = getattr(cur, '_path', [])
        print(f"Step {step}: {ph}, robot=({r['x']:.4f},{r['y']:.4f}), surf_x={surf_x:.4f}, place_y={place_y:.4f}, path_len={len(path)}")
        print(f"  block: bl=({b['x']:.4f},{b['y']:.4f}) w={b['width']:.4f} h={b['height']:.4f}")
        print(f"  surf: bl=({s['x']:.4f},{s['y']:.4f}) w={s['width']:.4f} h={s['height']:.4f}")
        if len(path) > 0:
            print(f"  path[0..min3]: {[f'({p[0]:.4f},{p[1]:.4f})' for p in path[:3]]}...{[f'({p[0]:.4f},{p[1]:.4f})' for p in path[-1:]]}")
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Done at step {step+1}!")
        break
