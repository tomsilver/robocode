import numpy as np
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction
from act_helpers import clip_action, toward_pos, normalize_angle, K_THETA, MAX_DTHETA

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)
print("obs0:", get_obstruction(obs, 0))
print("obs3:", get_obstruction(obs, 3))

# Navigate to x=0.465 but with theta=PI/2 (arm pointing UP)
# First navigate to safe position, then rotate arm up
for step in range(200):
    robot = get_robot(obs)
    # Move to x=0.465, y=0.80
    dx = np.clip((0.465 - robot['x']) * 3.0, -0.05, 0.05)
    dy = np.clip((0.80 - robot['y']) * 3.0, -0.05, 0.05)
    err = normalize_angle(np.pi/2 - robot['theta'])  # theta=pi/2 (arm UP)
    dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
    action = clip_action(dx=dx, dy=dy, dtheta=dtheta)
    obs, _, _, _, _ = env.step(action)
    robot = get_robot(obs)
    if abs(robot['x']-0.465)<0.01 and abs(robot['y']-0.80)<0.01 and abs(normalize_angle(np.pi/2-robot['theta']))<0.05:
        print(f"Ready: ({robot['x']:.3f},{robot['y']:.3f}), theta={robot['theta']:.3f}")
        break

# Now descend with arm pointing UP (theta=pi/2)
print("Descending with arm UP (theta=pi/2):")
prev_y = 999
for step in range(100):
    robot = get_robot(obs)
    if abs(robot['y']-prev_y)<0.001:
        print(f"Stuck at y={robot['y']:.5f} (arm still={robot['arm_joint']:.3f}, theta={robot['theta']:.3f})")
        break
    prev_y = robot['y']
    err = normalize_angle(np.pi/2 - robot['theta'])
    dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
    action = clip_action(dx=0.0, dy=-0.05, dtheta=dtheta)
    obs, _, _, _, _ = env.step(action)

# Now compare: navigate to x=0.465 with theta=-pi/2 (arm DOWN)  
env2 = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs2, info2 = env2.reset(seed=0)
for step in range(200):
    robot = get_robot(obs2)
    dx = np.clip((0.465 - robot['x']) * 3.0, -0.05, 0.05)
    dy = np.clip((0.80 - robot['y']) * 3.0, -0.05, 0.05)
    action = toward_pos(robot['x'], robot['y'], 0.465, 0.80, cur_theta=robot['theta'])
    obs2, _, _, _, _ = env2.step(action)
    robot = get_robot(obs2)
    if abs(robot['x']-0.465)<0.01 and abs(robot['y']-0.80)<0.01:
        print(f"\nReady arm-DOWN: ({robot['x']:.3f},{robot['y']:.3f}), theta={robot['theta']:.3f}")
        break

print("Descending with arm DOWN (theta=-pi/2):")
prev_y = 999
for step in range(100):
    robot = get_robot(obs2)
    if abs(robot['y']-prev_y)<0.001:
        print(f"Stuck at y={robot['y']:.5f} (arm={robot['arm_joint']:.3f}, theta={robot['theta']:.3f})")
        break
    prev_y = robot['y']
    action = toward_pos(robot['x'], robot['y'], 0.465, 0.0, cur_theta=robot['theta'])
    obs2, _, _, _, _ = env2.step(action)
