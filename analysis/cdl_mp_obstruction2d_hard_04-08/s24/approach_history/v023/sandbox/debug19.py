import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_obstruction, gripper_tip_pos

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
ob0_init = extract_obstruction(obs, 0)
print(f"Obs0: cx={ob0_init['cx']:.4f} cy={ob0_init['cy']:.4f} y2={ob0_init['y2']:.4f}")

# Navigate above obs0
target_x = ob0_init['cx']
target_y = ob0_init['y2'] + 0.21

for step in range(300):
    r = extract_robot(obs)
    dx = np.clip((target_x - r['x']), -0.05, 0.05)
    dy = np.clip((target_y - r['y']), -0.05, 0.05)
    dtheta = np.clip(((-np.pi/2 - r['theta'] + np.pi) % (2*np.pi) - np.pi)*2.0, -0.196, 0.196)
    obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, -0.10, 0], dtype=np.float32))
    r2 = extract_robot(obs)
    if abs(r2['x']-target_x)<0.005 and abs(r2['y']-target_y)<0.005 and abs(r2['arm_joint']-0.10)<0.005:
        break

# Extend arm
for step in range(3):
    obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0.10, 1.0], dtype=np.float32))

# Now move robot left and see if obs0 follows
print("Moving robot left with vacuum=1:")
for step in range(10):
    r = extract_robot(obs)
    obs, _, _, _, _ = env.step(np.array([-0.04, 0, 0, 0, 1.0], dtype=np.float32))
    r2 = extract_robot(obs)
    ob2 = extract_obstruction(obs, 0)
    print(f"  step={step}: robot_x={r2['x']:.4f} obs_cx={ob2['cx']:.4f} (delta={ob2['cx']-ob0_init['cx']:.4f})")
    if abs(ob2['cx'] - ob0_init['cx']) > 0.01:
        print("  *** OBS0 IS HELD! ***")
        break
