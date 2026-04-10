import numpy as np
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, is_vacuum_on, obstruction_overlaps_surface
from act_helpers import clip_action, toward_pos, normalize_angle, K_THETA, MAX_DTHETA

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

obs0 = get_obstruction(obs, 0)
print(f"obs0: {obs0}")
obs0_top = obs0['y'] + obs0['height']/2
print(f"obs0_top={obs0_top:.4f}")

# Navigate to obs0 x, descend until stuck  
target_x = obs0['x']
for step in range(200):
    robot = get_robot(obs)
    action = toward_pos(robot['x'], robot['y'], target_x, 0.70, cur_theta=robot['theta'])
    obs, _, _, _, _ = env.step(action)
    robot = get_robot(obs)
    if abs(robot['x']-target_x)<0.01 and abs(robot['y']-0.70)<0.01:
        break

# Descend until stuck
prev_y = 999
for step in range(100):
    robot = get_robot(obs)
    if abs(robot['y']-prev_y)<0.001:
        print(f"Stuck at y={robot['y']:.4f}, arm={robot['arm_joint']:.4f}")
        break
    prev_y = robot['y']
    action = clip_action(dy=-0.05)
    obs, _, _, _, _ = env.step(action)

robot = get_robot(obs)
stuck_y = robot['y']

# Now extend arm to pick obs0
arm_needed = stuck_y - obs0_top - 0.015
print(f"arm_joint_needed={arm_needed:.4f}")

# Extend arm
for step in range(50):
    robot = get_robot(obs)
    darm = np.clip((arm_needed - robot['arm_joint'])*8.0, -0.1, 0.1)
    if abs(robot['arm_joint']-arm_needed)<0.005:
        print(f"Arm at {robot['arm_joint']:.4f}, suction_y={robot['y']-(robot['arm_joint']+0.015):.4f}")
        break
    obs, _, _, _, _ = env.step(clip_action(darm=darm))

# Apply vacuum and check if grasped
print("Applying vacuum:")
for step in range(20):
    robot = get_robot(obs)
    action = clip_action(vac=1.0)
    obs, _, _, _, _ = env.step(action)
    if step % 5 == 0 or step == 19:
        o0_still_on = obstruction_overlaps_surface(obs, 0)
        vac = robot['vacuum']
        print(f"  step {step}: vac={vac:.2f} obs0_on_surface={o0_still_on}")

# Try to retract and move
print("\nAttempting retract + lift:")
for step in range(20):
    robot = get_robot(obs)
    darm = np.clip((0.10 - robot['arm_joint'])*8.0, -0.1, 0.1)
    action = clip_action(darm=darm, vac=1.0)
    obs, _, _, _, _ = env.step(action)
    robot2 = get_robot(obs)
    print(f"  step {step}: arm={robot2['arm_joint']:.3f}, y={robot2['y']:.4f}, vac={robot2['vacuum']:.2f}")
