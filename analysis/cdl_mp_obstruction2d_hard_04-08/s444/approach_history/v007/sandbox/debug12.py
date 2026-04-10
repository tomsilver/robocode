import numpy as np
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction
from act_helpers import clip_action, toward_pos

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

# Navigate to (0.300, 0.2808) first 
for step in range(200):
    robot = get_robot(obs)
    action = toward_pos(robot['x'], robot['y'], 0.300, 0.50, cur_theta=robot['theta'])
    obs, _, _, _, _ = env.step(action)
    robot = get_robot(obs)
    if abs(robot['x']-0.300)<0.005 and abs(robot['y']-0.50)<0.005:
        break

# Navigate to y=0.2808 at x=0.300 
prev_y = None
for step in range(60):
    robot = get_robot(obs)
    if prev_y and abs(robot['y']-prev_y)<0.001:
        print(f"Reached min at y={robot['y']:.5f}")
        break
    prev_y = robot['y']
    action = clip_action(dx=0.0, dy=-0.05)
    obs, _, _, _, _ = env.step(action)

robot = get_robot(obs)
print(f"Position: ({robot['x']:.4f},{robot['y']:.4f})")

# Now slide RIGHT to see when we hit obs0 (left edge at 0.434)
print("Sliding RIGHT:")
prev_x = None
for step in range(50):
    robot = get_robot(obs)
    if step % 5 == 0 or (prev_x and abs(robot['x']-prev_x)<0.001):
        stuck = prev_x and abs(robot['x']-prev_x)<0.001
        print(f"  step {step}: ({robot['x']:.4f},{robot['y']:.4f}) {'STUCK' if stuck else ''}")
        if stuck:
            break
    prev_x = robot['x']
    action = clip_action(dx=0.05, dy=0.0)
    obs, _, _, _, _ = env.step(action)

# Verify obs0 left edge
print(f"\nobs0 x_range: [{0.4912-0.1144/2:.4f}, {0.4912+0.1144/2:.4f}]")
print(f"obs0 y_range: [{0.1000-0.1219/2:.4f}, {0.1000+0.1219/2:.4f}]")
