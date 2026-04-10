import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_target_surface, extract_target_block

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}
obs, info = env.reset(seed=1)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

in_navplace = False
nav_count = 0
for step in range(2000):
    cur = approach._current
    bname = type(cur).__name__
    ph = getattr(cur, '_phase', '?')
    
    if bname == 'PlaceTargetBlock' and ph == 'NAV_PLACE' and not in_navplace:
        in_navplace = True
        print(f"Step {step}: Entered NAV_PLACE")
        r = extract_robot(obs)
        print(f"  robot: x={r['x']:.4f} y={r['y']:.4f}")
        b = extract_target_block(obs)
        print(f"  block: x={b['x']:.4f} y={b['y']:.4f} w={b['width']:.4f} h={b['height']:.4f}")
        s = extract_target_surface(obs)
        print(f"  surf:  x={s['x']:.4f} y={s['y']:.4f} w={s['width']:.4f} h={s['height']:.4f}")
        pl = approach._current
        print(f"  _surf_x={getattr(pl,'_surf_x','?'):.4f} _place_y={getattr(pl,'_place_y','?'):.4f}")
        print(f"  _path len={len(getattr(pl,'_path',[]))}")
        print(f"  _path_step={getattr(pl,'_path_step','?')}")
        
    if in_navplace:
        nav_count += 1
        if nav_count % 100 == 0:
            r = extract_robot(obs)
            pl = approach._current
            print(f"  nav_count={nav_count}: robot=({r['x']:.4f},{r['y']:.4f}), path_step={getattr(pl,'_path_step','?')}/{len(getattr(pl,'_path',[]))}")
    
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Done at step {step+1}")
        break

print(f"Total nav_place steps: {nav_count}")
