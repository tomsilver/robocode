import numpy as np
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction
from act_helpers import clip_action, toward_pos

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

# Get robot stuck at (0.465, 0.3308)
for step in range(200):
    robot = get_robot(obs)
    action = toward_pos(robot['x'], robot['y'], 0.465, 0.50, cur_theta=robot['theta'])
    obs, _, _, _, _ = env.step(action)
    robot = get_robot(obs)
    if abs(robot['x']-0.465)<0.01 and abs(robot['y']-0.50)<0.01:
        break

prev_y = None
for step in range(60):
    robot = get_robot(obs)
    if prev_y and abs(robot['y']-prev_y)<0.001:
        break
    prev_y = robot['y']
    action = clip_action(dx=0.0, dy=-0.05)
    obs, _, _, _, _ = env.step(action)

robot = get_robot(obs)
print(f"Stuck at: ({robot['x']:.4f},{robot['y']:.4f})")

# Now try moving in different directions
directions = [
    ('LEFT', (-0.05, 0.0)),
    ('RIGHT', (0.05, 0.0)),
    ('DOWN-LEFT', (-0.05, -0.05)),
    ('DOWN-RIGHT', (0.05, -0.05)),
]
for name, (dx, dy) in directions:
    env2 = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs2, info2 = env2.reset(seed=0)
    for step in range(200):
        robot = get_robot(obs2)
        action = toward_pos(robot['x'], robot['y'], 0.465, 0.50, cur_theta=robot['theta'])
        obs2, _, _, _, _ = env2.step(action)
        robot = get_robot(obs2)
        if abs(robot['x']-0.465)<0.01 and abs(robot['y']-0.50)<0.01:
            break
    prev_y2 = None
    for step in range(60):
        robot = get_robot(obs2)
        if prev_y2 and abs(robot['y']-prev_y2)<0.001:
            break
        prev_y2 = robot['y']
        action = clip_action(dx=0.0, dy=-0.05)
        obs2, _, _, _, _ = env2.step(action)
    # Now move in direction
    before = get_robot(obs2)
    action = clip_action(dx=dx, dy=dy)
    obs2, _, _, _, _ = env2.step(action)
    after = get_robot(obs2)
    moved_x = abs(after['x']-before['x']) > 0.001
    moved_y = abs(after['y']-before['y']) > 0.001
    print(f"{name}: ({before['x']:.4f},{before['y']:.4f}) → ({after['x']:.4f},{after['y']:.4f}) {'moved' if moved_x or moved_y else 'STUCK'}")
