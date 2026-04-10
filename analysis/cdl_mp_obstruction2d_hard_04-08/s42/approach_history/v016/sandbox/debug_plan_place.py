import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_target_surface, extract_target_block, extract_obstruction, NUM_OBSTRUCTIONS
from act_helpers import plan_base_path, make_collision_fn, TABLE_HEIGHT, COLLISION_MARGIN

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}
obs, info = env.reset(seed=1)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

plan_place_hit = False
for step in range(300):
    cur = approach._current
    bname = type(cur).__name__
    ph = getattr(cur, '_phase', '?')
    
    if bname == 'PlaceTargetBlock' and ph == 'PLAN_PLACE' and not plan_place_hit:
        plan_place_hit = True
        print(f"Step {step}: PLAN_PLACE")
        r = extract_robot(obs)
        print(f"  robot: x={r['x']:.4f} y={r['y']:.4f}")
        surf_x = getattr(cur, '_surf_x', '?')
        place_y = getattr(cur, '_place_y', '?')
        print(f"  surf_x={surf_x:.4f} place_y={place_y:.4f}")
        base_r = r['base_radius']
        print(f"  Obstructions:")
        for i in range(NUM_OBSTRUCTIONS):
            ob = extract_obstruction(obs, i)
            cx = ob['x'] + ob['width']/2
            cy = ob['y'] + ob['height']/2
            print(f"    obs[{i}]: bl=({ob['x']:.4f},{ob['y']:.4f}) w={ob['width']:.4f} h={ob['height']:.4f} center=({cx:.4f},{cy:.4f})")
        # Test collision at surf_x, place_y
        obs_rects = []
        for i in range(NUM_OBSTRUCTIONS):
            ob = extract_obstruction(obs, i)
            cx = ob['x'] + ob['width']/2
            cy = ob['y'] + ob['height']/2
            obs_rects.append((cx, cy, ob['width'], ob['height']))
        cfn = make_collision_fn(base_r, obs_rects, TABLE_HEIGHT + base_r)
        goal = np.array([surf_x, place_y])
        print(f"  collision at goal ({surf_x:.4f},{place_y:.4f}): {cfn(goal)}")
        # Test various x values
        for x_test in [0.80, 0.82, 0.84, 0.846, 0.86, 0.88, 0.90]:
            state = np.array([x_test, place_y])
            coll = cfn(state)
            print(f"    x={x_test:.3f} at place_y: collision={coll}")
    
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Done at step {step+1}")
        break
