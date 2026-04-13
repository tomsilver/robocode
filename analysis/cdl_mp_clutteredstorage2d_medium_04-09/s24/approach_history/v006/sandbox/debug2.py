import sys, os
sys.path.insert(0, '/sandbox')
os.chdir('/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach
from obs_helpers import get_blocks_outside_shelf, extract_robot, is_block_in_shelf, BLOCK_NAMES, get_block_center, get_shelf_slot
from primitives.motion_planning import BiRRT
PRIMITIVES = {'BiRRT': BiRRT}

for seed in [0, 1]:
    print(f"\n=== SEED {seed} ===")
    env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, PRIMITIVES)
    approach.reset(obs, info)
    
    prev_phase = None
    for step in range(600):
        cur = approach._current
        phase = getattr(cur, '_phase', None)
        btype = type(cur).__name__[:5]
        
        r = extract_robot(obs)
        
        if btype == 'Place':
            hb = getattr(cur, '_held_block', None)
            if phase == 2:  # EXTEND
                bcx, bcy = (get_block_center(obs, hb) if hb else (0,0))
                in_shelf = is_block_in_shelf(obs, hb) if hb else False
                slot = get_shelf_slot(obs)
                if step % 10 == 0:
                    print(f"  s{step:4d} ph=EXTEND r=({r.x:.3f},{r.y:.3f},{r.theta:.3f}) arm={r.arm_joint:.3f} b=({bcx:.3f},{bcy:.3f}) in={in_shelf} slot={slot[0]:.3f}+{slot[2]:.3f}")
            elif phase == 1:  # ORIENT
                hb_val = hb
                if hb_val:
                    bcx, bcy = get_block_center(obs, hb_val)
                    block_offset_x = bcx - r.x
                    slot = get_shelf_slot(obs)
                    slot_cx = slot[0] + slot[2]/2
                    adj_x = slot_cx - block_offset_x
                    from obs_helpers import ROBOT_BASE_RADIUS
                    adj_x_clamped = max(ROBOT_BASE_RADIUS + 0.005, adj_x)
                    if step % 10 == 0:
                        print(f"  s{step:4d} ph=ORIENT r=({r.x:.3f},{r.y:.3f}) b=({bcx:.3f},{bcy:.3f}) off_x={block_offset_x:.3f} adj_x={adj_x:.3f}→{adj_x_clamped:.3f}")
        
        if btype == 'Picku' and phase == 0 and step % 30 == 0:
            bname = getattr(cur, '_block_name', None)
            path_idx = getattr(cur, '_path_idx', None)
            path_len = len(getattr(cur, '_path', []))
            print(f"  s{step:4d} ph=NAV r=({r.x:.3f},{r.y:.3f}) path_idx={path_idx}/{path_len}")
        
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"  SUCCESS at step {step+1}")
            break
    else:
        print(f"  FAILED: cur={type(approach._current).__name__} ph={getattr(approach._current,'_phase',None)}")
