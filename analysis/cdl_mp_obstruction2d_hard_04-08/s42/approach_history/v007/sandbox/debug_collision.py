import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import *

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, _ = env.reset(seed=42)
block = extract_target_block(obs)
robot = extract_robot(obs)
bx, by = block['x'], block['y']
print(f"Block: ({bx:.4f},{by:.4f}) w={block['width']:.4f} h={block['height']:.4f}")
print(f"Block occupies y=[{by - block['height']/2:.4f}, {by + block['height']/2:.4f}]")
print()

# Test what y the robot can reach directly above the block
# First navigate robot to (bx, 0.75) then try to descend
for step in range(40):  
    act = np.zeros(5)
    act[0] = np.clip(bx - extract_robot(obs)['x'], -0.05, 0.05)
    act[1] = np.clip(0.75 - extract_robot(obs)['y'], -0.05, 0.05)
    obs, _, _, _, _ = env.step(act)

robot = extract_robot(obs)
print(f"After approach: robot=({robot['x']:.4f},{robot['y']:.4f})")

# Now descend
print("Descent test (arm retracted):")
for step in range(30):
    act = np.zeros(5); act[1] = -0.05
    prev_y = extract_robot(obs)['y']
    obs, _, _, _, _ = env.step(act)
    new_y = extract_robot(obs)['y']
    stuck = abs(prev_y - new_y) < 0.001
    print(f"  y={prev_y:.4f} -> {new_y:.4f} {'STUCK' if stuck else ''}")
    if stuck:
        print(f"  Robot bottom = {new_y - extract_robot(obs)['base_radius']:.4f}, block top = {by + block['height']/2:.4f}")
        break
