import sys, os
sys.path.insert(0, '/sandbox')
os.chdir('/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach
from obs_helpers import get_blocks_outside_shelf, extract_robot, is_block_in_shelf, BLOCK_NAMES, get_block_center
from primitives.motion_planning import BiRRT
PRIMITIVES = {'BiRRT': BiRRT}

for seed in [0, 1]:
    print(f"\n=== SEED {seed} ===")
    env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, PRIMITIVES)
    approach.reset(obs, info)
    
    print(f"  Behaviors: {[type(b).__name__ for b in list(approach._behaviors)]}")
    print(f"  Current: {type(approach._current).__name__}")
    outside = get_blocks_outside_shelf(obs)
    print(f"  Outside: {outside}")
    r = extract_robot(obs)
    print(f"  Robot: ({r.x:.3f},{r.y:.3f}) theta={r.theta:.3f} arm={r.arm_joint:.3f} vac={r.vacuum:.1f}")
    for name in BLOCK_NAMES:
        cx,cy = get_block_center(obs, name)
        print(f"  {name}: ({cx:.3f},{cy:.3f}) in_shelf={is_block_in_shelf(obs,name)}")
    
    # Run 500 steps with periodic trace
    for step in range(500):
        cur = approach._current
        b_name = getattr(cur, '_block_name', None)
        phase = getattr(cur, '_phase', None)
        
        if step % 50 == 0:
            r = extract_robot(obs)
            btype = type(cur).__name__[:5]
            print(f"  s{step:4d} {btype} ph={phase} r=({r.x:.2f},{r.y:.2f}) arm={r.arm_joint:.3f} vac={r.vacuum:.1f}", end="")
            if b_name:
                cx,cy = get_block_center(obs, b_name)
                print(f" {b_name}=({cx:.2f},{cy:.2f})", end="")
            print()
        
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"  SUCCESS at step {step+1}")
            break
    else:
        r = extract_robot(obs)
        outside = get_blocks_outside_shelf(obs)
        cur = approach._current
        print(f"  FAILED: outside={outside} cur_phase={getattr(cur,'_phase',None)} r=({r.x:.3f},{r.y:.3f})")
