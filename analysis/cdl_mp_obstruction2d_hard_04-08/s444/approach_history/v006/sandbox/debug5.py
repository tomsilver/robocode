import numpy as np
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from motion_planning import BiRRT
from obs_helpers import get_robot, get_obstruction
from act_helpers import clip_action, toward_pos

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

# Manually navigate to (0.491, 0.50) very precisely, then try going to (0.491, 0.301)
target_x, nav_y = 0.491, 0.50
pick_y = 0.100 + 0.122/2 + 0.14  # obs0 pick_y

print(f"Navigating to ({target_x:.3f}, {nav_y:.3f}), then pick_y={pick_y:.3f}")

# Phase 1: navigate to (target_x, nav_y)
for step in range(50):
    robot = get_robot(obs)
    action = toward_pos(robot['x'], robot['y'], target_x, nav_y, cur_theta=robot['theta'])
    obs, _, _, _, _ = env.step(action)
    robot = get_robot(obs)
    if abs(robot['x']-target_x)<0.005 and abs(robot['y']-nav_y)<0.005:
        print(f"Reached nav_high at step {step}: ({robot['x']:.4f},{robot['y']:.4f})")
        break

# Phase 2: navigate down to pick_y (straight down)
print(f"\nDescending to pick_y={pick_y:.3f}:")
for step in range(60):
    robot = get_robot(obs)
    action = toward_pos(robot['x'], robot['y'], target_x, pick_y, cur_theta=robot['theta'])
    obs, _, _, _, _ = env.step(action)
    robot2 = get_robot(obs)
    if step % 5 == 0 or abs(robot2['y']-robot['y'])<0.001:
        print(f"  step {step}: ({robot['x']:.4f},{robot['y']:.4f}) → ({robot2['x']:.4f},{robot2['y']:.4f}) {'STUCK' if abs(robot2['y']-robot['y'])<0.001 else ''}")
    if abs(robot2['y']-pick_y)<0.005:
        print(f"Reached pick_y!")
        break
    if abs(robot2['y']-robot['y'])<0.001:
        print(f"Stuck at y={robot2['y']:.4f}")
        break
