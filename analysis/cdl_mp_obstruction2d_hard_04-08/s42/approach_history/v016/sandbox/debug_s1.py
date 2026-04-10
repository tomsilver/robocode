import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_target_block, extract_target_surface, extract_obstruction, NUM_OBSTRUCTIONS
from act_helpers import make_carry_collision_fn, TABLE_HEIGHT, NAV_HEIGHT

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}
obs, info = env.reset(seed=1)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

done = False
steps = 0
last_phase = None

while not done and steps < 2000:
    cur = approach._current
    phase = getattr(cur, '_phase', 'N/A')
    
    if type(cur).__name__ == 'PlaceTargetBlock' and phase != last_phase:
        r = extract_robot(obs)
        blk = extract_target_block(obs)
        surf = extract_target_surface(obs)
        print(f"  step={steps} phase={phase} r=({r['x']:.4f},{r['y']:.4f},{r['arm_joint']:.3f}) vac={r['vacuum']:.0f}")
        print(f"  surf=({surf['x']:.4f},{surf['y']:.4f},{surf['width']:.4f},{surf['height']:.4f}) surf_cx={surf['x']+surf['width']/2:.4f}")
        print(f"  blk=({blk['x']:.4f},{blk['y']:.4f},{blk['width']:.4f},{blk['height']:.4f}) carry_half={blk['width']/2:.4f}")
        print(f"  self._surf_x={getattr(cur,'_surf_x','N/A'):.4f} self._place_y={getattr(cur,'_place_y','N/A'):.4f}")
        
        # Check carry collision at planned surf_x
        if hasattr(cur, '_surf_x'):
            surf_x = cur._surf_x
            r2 = extract_robot(obs)
            obs_rects = []
            for i in range(NUM_OBSTRUCTIONS):
                ob = extract_obstruction(obs, i)
                cx = ob['x']+ob['width']/2; cy = ob['y']+ob['height']/2
                obs_rects.append((cx, cy, ob['width'], ob['height']))
            cfn = make_carry_collision_fn(r2['base_radius'], obs_rects, blk['width'], blk['height'], TABLE_HEIGHT+r2['base_radius'])
            state = np.array([surf_x, NAV_HEIGHT])
            print(f"  carry_cfn at surf_x={surf_x:.4f},NAV_HEIGHT: collision={cfn(state)}")
            for i in range(NUM_OBSTRUCTIONS):
                ob = extract_obstruction(obs, i)
                print(f"    obs[{i}]=({ob['x']:.3f},{ob['y']:.3f},{ob['width']:.3f},{ob['height']:.3f}) cy={ob['y']+ob['height']/2:.3f}")
        last_phase = phase

    if type(cur).__name__ == 'PlaceTargetBlock' and phase == 'NAV_SURF' and steps > 200:
        r = extract_robot(obs)
        blk = extract_target_block(obs)
        print(f"  step={steps} NAV_SURF r=({r['x']:.4f},{r['y']:.4f}) surf_x={getattr(cur,'_surf_x','?'):.4f} dist={abs(r['x']-getattr(cur,'_surf_x',0)):.4f}")
        if steps > 400:
            break

    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    steps += 1

print(f"Done: steps={steps} terminated={terminated if done else 'N/A'}")
