import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_obstruction, gripper_tip_pos

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
ob0 = extract_obstruction(obs, 0)
print(f"Obs0: x=[{ob0['x1']:.4f},{ob0['x2']:.4f}] y=[{ob0['y1']:.4f},{ob0['y2']:.4f}] cx={ob0['cx']:.4f}")

# Navigate DIRECTLY ON TOP of obs0 center (robot center at obs0 center!)
# This ensures suction deeply overlaps obs0
target_x = ob0['cx']
target_y = ob0['y2'] + 0.30  # well above, arm pointing down
for step in range(300):
    r = extract_robot(obs)
    dx = np.clip((target_x - r['x']), -0.05, 0.05)
    dy = np.clip((target_y - r['y']), -0.05, 0.05)
    dtheta = np.clip(((-np.pi/2 - r['theta'] + np.pi) % (2*np.pi) - np.pi)*2.0, -0.196, 0.196)
    obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, -0.10, 0], dtype=np.float32))
    r2 = extract_robot(obs)
    if abs(r2['x']-target_x)<0.005 and abs(r2['y']-target_y)<0.005 and abs(r2['arm_joint']-0.10)<0.005:
        break

r = extract_robot(obs)
tip = gripper_tip_pos(obs)
print(f"Robot: ({r['x']:.4f},{r['y']:.4f}) theta={r['theta']:.4f} arm={r['arm_joint']:.4f}")
print(f"Suction center at min arm: {tip}")

# Now try various arm extensions and vacuum on
print("\nExtending arm (delta=0.02 each step, v=1 always):")
for step in range(15):
    r = extract_robot(obs)
    tip = gripper_tip_pos(obs)
    obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0.02, 1.0], dtype=np.float32))
    r2 = extract_robot(obs)
    ob2 = extract_obstruction(obs, 0)
    moved = abs(ob2['cx'] - ob0['cx']) > 0.001 or abs(ob2['cy'] - ob0['cy']) > 0.001
    print(f"  step={step}: arm {r['arm_joint']:.4f}->{r2['arm_joint']:.4f} suc_y={tip[1]:.4f} obs_y2={ob0['y2']:.4f} MOVED={moved}")
    if moved:
        print("  *** GRASPED! ***")
        break
