import numpy as np
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction
from act_helpers import clip_action, toward_pos

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

# Navigate to a completely clear area (x=0.2, no obstructions nearby)
print("Navigating to clear area x=0.20:")
for step in range(100):
    robot = get_robot(obs)
    action = toward_pos(robot['x'], robot['y'], 0.20, 0.50, cur_theta=robot['theta'])
    obs, _, _, _, _ = env.step(action)
    robot = get_robot(obs)
    if abs(robot['x']-0.20)<0.005 and abs(robot['y']-0.50)<0.005:
        print(f"Reached ({robot['x']:.4f},{robot['y']:.4f})")
        break

# Now try going VERY LOW
print("Descending as low as possible:")
min_y = float('inf')
prev_y = None
for step in range(100):
    robot = get_robot(obs)
    if prev_y is not None and abs(robot['y']-prev_y)<0.0005:
        print(f"Stuck at y={robot['y']:.5f}")
        break
    prev_y = robot['y']
    action = clip_action(dx=0.0, dy=-0.05, dtheta=0.0, darm=0.0, vac=0.0)
    obs, _, _, _, _ = env.step(action)
    robot2 = get_robot(obs)
    if step % 5 == 0:
        print(f"  step {step}: y={robot2['y']:.4f}")
    min_y = min(min_y, robot2['y'])

print(f"\nMinimum y reached: {min_y:.5f}")
print(f"At this y, arm shaft bottom = {min_y - 0.10:.5f}")
print(f"obs0 top = {0.100 + 0.122/2:.4f}")
print(f"Table top = {0.05:.4f}")
