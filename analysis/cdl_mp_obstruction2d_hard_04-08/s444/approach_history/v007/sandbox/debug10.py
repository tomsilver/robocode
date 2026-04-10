import numpy as np
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction
from act_helpers import clip_action, toward_pos

def find_min_y(env, obs, target_x):
    """Navigate to target_x and find min y."""
    for step in range(100):
        robot = get_robot(obs)
        action = toward_pos(robot['x'], robot['y'], target_x, 0.80, cur_theta=robot['theta'])
        obs, _, _, _, _ = env.step(action)
        robot = get_robot(obs)
        if abs(robot['x']-target_x)<0.005 and abs(robot['y']-0.80)<0.005:
            break
    # Now descend
    prev_y = 999
    for step in range(200):
        robot = get_robot(obs)
        if abs(robot['y']-prev_y)<0.0001:
            return robot['y'], obs
        prev_y = robot['y']
        action = clip_action(dx=0.0, dy=-0.05, dtheta=0.0, darm=0.0, vac=0.0)
        obs, _, _, _, _ = env.step(action)
    return prev_y, obs

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)
print("obs0:", get_obstruction(obs, 0))  
print("obs3:", get_obstruction(obs, 3))

# Test multiple x positions
test_xs = [0.20, 0.30, 0.40, 0.434, 0.465, 0.491, 0.520, 0.548, 0.60, 0.70]
for tx in test_xs:
    env2 = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs2, info2 = env2.reset(seed=0)
    min_y, _ = find_min_y(env2, obs2, tx)
    robot_bottom = min_y - 0.10
    print(f"x={tx:.3f}: min_y={min_y:.4f}, robot_bottom={robot_bottom:.4f}")
