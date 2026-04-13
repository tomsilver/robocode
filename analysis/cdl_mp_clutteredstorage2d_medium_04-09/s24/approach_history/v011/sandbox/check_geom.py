import sys, os
sys.path.insert(0, '/sandbox')
os.chdir('/sandbox')
import numpy as np
import math
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import *
from primitives.motion_planning import BiRRT
PRIMITIVES = {'BiRRT': BiRRT}

for seed in [0, 1]:
    print(f"\n=== SEED {seed} ===")
    env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
    obs, info = env.reset(seed=seed)
    
    slot = get_shelf_slot(obs)
    print(f"  shelf slot: x1={slot[0]:.3f} y1={slot[1]:.3f} w1={slot[2]:.3f} h1={slot[3]:.3f}")
    
    r = extract_robot(obs)
    print(f"  robot: ({r.x:.3f},{r.y:.3f}) theta={r.theta:.3f}")
    
    for name in BLOCK_NAMES:
        base, _ = LAYOUT[name]
        p = extract_rect(obs, name)
        cx, cy = get_block_center(obs, name)
        print(f"  {name}: corner=({p.x:.3f},{p.y:.3f}) theta={p.theta:.3f} center=({cx:.3f},{cy:.3f}) in={is_block_in_shelf(obs, name)}")
    
    # Simulate placing block1: robot at (0.205, 2.100), arm=0.500, theta=pi/2
    # After release, where does block settle?
    # Create a fake obs with block1 in shelf position
    print("  If placed at arm=0.500: block center y =", 2.100 + 0.500 + 0.03)
    print("  If placed at arm=0.550: block center y =", 2.100 + 0.550 + 0.03)

