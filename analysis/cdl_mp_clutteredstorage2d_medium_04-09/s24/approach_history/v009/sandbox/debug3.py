import sys, os
sys.path.insert(0, '/sandbox')
os.chdir('/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach
from obs_helpers import get_blocks_outside_shelf, extract_robot, is_block_in_shelf, BLOCK_NAMES, get_block_center, get_shelf_slot, suction_center_pos
from primitives.motion_planning import BiRRT
PRIMITIVES = {'BiRRT': BiRRT}

for seed in [0, 1, 2, 42]:
    print(f"\n=== SEED {seed} ===")
    env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, PRIMITIVES)
    approach.reset(obs, info)
    
    prev_btype = None
    prev_phase = None
    for step in range(2000):
        cur = approach._current
        btype = type(cur).__name__[:5]
        phase = getattr(cur, '_phase', None)
        r = extract_robot(obs)
        
        if btype != prev_btype or phase != prev_phase:
            hb = getattr(cur, '_block_name', getattr(cur, '_held_block', None))
            sx, sy = suction_center_pos(r)
            if hb:
                bcx, bcy = get_block_center(obs, hb)
                suct_dist = ((sx-bcx)**2+(sy-bcy)**2)**0.5
            else:
                suct_dist = -1
            print(f"  s{step:5d} {btype} ph={phase} r=({r.x:.3f},{r.y:.3f}) arm={r.arm_joint:.3f} vac={r.vacuum:.1f} hb={hb} sd={suct_dist:.3f}")
            prev_btype = btype
            prev_phase = phase
        
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"  SUCCESS at step {step+1}")
            break
    else:
        r = extract_robot(obs)
        cur = approach._current
        phase = getattr(cur, '_phase', None)
        hb = getattr(cur, '_block_name', getattr(cur, '_held_block', None))
        outside = get_blocks_outside_shelf(obs)
        sx, sy = suction_center_pos(r)
        if hb:
            bcx, bcy = get_block_center(obs, hb)
            suct_dist = ((sx-bcx)**2+(sy-bcy)**2)**0.5
        else:
            suct_dist = -1
        print(f"  FAIL: outside={outside} ph={phase} hb={hb} sd={suct_dist:.3f} r=({r.x:.3f},{r.y:.3f}) arm={r.arm_joint:.3f}")
