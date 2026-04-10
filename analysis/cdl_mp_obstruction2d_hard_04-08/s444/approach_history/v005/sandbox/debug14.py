import numpy as np
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction
from act_helpers import clip_action, toward_pos

def find_min_y_with_arm(target_x, target_arm, seed=0):
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=seed)
    # Navigate to target_x, y=0.80
    for step in range(200):
        robot = get_robot(obs)
        action = toward_pos(robot['x'], robot['y'], target_x, 0.80, cur_theta=robot['theta'])
        obs, _, _, _, _ = env.step(action)
        robot = get_robot(obs)
        if abs(robot['x']-target_x)<0.01 and abs(robot['y']-0.80)<0.01:
            break
    # Set arm to target_arm
    for step in range(50):
        robot = get_robot(obs)
        darm = np.clip((target_arm - robot['arm_joint'])*8.0, -0.1, 0.1)
        obs, _, _, _, _ = env.step(clip_action(darm=darm))
    # Descend
    prev_y = 999
    for step in range(200):
        robot = get_robot(obs)
        if abs(robot['y']-prev_y)<0.001:
            return robot['y'], robot['arm_joint']
        prev_y = robot['y']
        obs, _, _, _, _ = env.step(clip_action(dy=-0.05))
    return prev_y, get_robot(obs)['arm_joint']

# Test obs0 position (x=0.491) with different arm lengths
print("At x=0.491 (obs0_x), different arm_joint values:")
for arm in [0.10, 0.13, 0.15, 0.18, 0.20]:
    min_y, actual_arm = find_min_y_with_arm(0.491, arm)
    obs0_top = 0.100 + 0.122/2
    print(f"  arm_joint={arm:.2f} (actual={actual_arm:.3f}): min_y={min_y:.4f}, arm_bottom={min_y-actual_arm:.4f}, gap_to_obs0={min_y-actual_arm-obs0_top:.4f}")

print("\nAt x=0.20 (clear), different arm_joint values:")
for arm in [0.10, 0.13, 0.20]:
    min_y, actual_arm = find_min_y_with_arm(0.20, arm)
    print(f"  arm_joint={arm:.2f} (actual={actual_arm:.3f}): min_y={min_y:.4f}, arm_bottom={min_y-actual_arm:.4f}")
