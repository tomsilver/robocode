import numpy as np
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction
from act_helpers import clip_action, toward_pos

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

target_x = 0.491
pick_y = 0.100 + 0.122/2 + 0.14  # 0.301

# Navigate tightly to (target_x, 0.50)
for step in range(100):
    robot = get_robot(obs)
    action = toward_pos(robot['x'], robot['y'], target_x, 0.50, cur_theta=robot['theta'])
    obs, _, _, _, _ = env.step(action)
    robot = get_robot(obs)
    if abs(robot['x']-target_x)<0.002 and abs(robot['y']-0.50)<0.005:
        print(f"Precisely at nav_high: ({robot['x']:.5f}, {robot['y']:.5f})")
        break

print(f"Gripper right edge: {robot['x']+0.035:.4f}")
print(f"Obs3 left edge: {0.5355:.4f}")

# Now descend STRAIGHT DOWN (no x movement)
print("Descending straight down:")
for step in range(50):
    robot = get_robot(obs)
    prev_y = robot['y']
    action = clip_action(dx=0.0, dy=-0.05, dtheta=0.0, darm=0.0, vac=0.0)
    obs, _, _, _, _ = env.step(action)
    robot2 = get_robot(obs)
    print(f"  step {step}: ({robot2['x']:.5f},{robot2['y']:.5f}) {'STUCK' if abs(robot2['y']-prev_y)<0.001 else ''}")
    if abs(robot2['y']-pick_y)<0.01:
        print("Reached pick_y!")
        break
    if abs(robot2['y']-prev_y)<0.001:
        print(f"Stuck at y={robot2['y']:.5f}")
        break
