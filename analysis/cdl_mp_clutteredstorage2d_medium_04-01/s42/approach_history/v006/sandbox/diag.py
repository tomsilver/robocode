import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import *

env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')

for seed in [0, 1, 2, 3, 42]:
    obs, info = env.reset(seed=seed)
    robot = extract_robot(obs)
    sx_min, sx_max, sy_min, sy_max = shelf_inner_bounds(obs)
    sy_bottom = shelf_y_bottom(obs)
    
    print(f"Seed {seed}:")
    print(f"  shelf inner: x=[{sx_min:.3f},{sx_max:.3f}] y=[{sy_min:.3f},{sy_max:.3f}]")
    print(f"  shelf y_bottom (outer): {sy_bottom:.3f}")
    print(f"  robot: x={robot.x:.3f}, y={robot.y:.3f}")
    
    for bn in BLOCK_NAMES:
        rect = extract_rect(obs, bn)
        cx, cy = block_center(rect)
        in_shelf = is_block_in_shelf(obs, bn)
        print(f"  {bn}: theta={rect.theta:.4f} center=({cx:.3f},{cy:.3f}) in_shelf={in_shelf}")
    print()
