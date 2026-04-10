import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_obstruction

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)

ob = extract_obstruction(obs, 0)
print(f"Obs0: x1={ob['x1']:.4f} cx={ob['cx']:.4f}")

# Target position: robot_x = obj_x1 - 0.035 - 0.195 = obj_x1 - 0.230
target_x = ob['x1'] - 0.230
target_y = max(0.22, ob['cy'])
print(f"Target robot: ({target_x:.4f},{target_y:.4f})")
print(f"  arm_joint_contact = {ob['x1'] - target_x - 0.035:.4f} (need >= 0.16)")
print(f"  full_ext penetration = {target_x + 0.235 - ob['x1']:.4f} (need <= 0.005)")

# Navigate there
for step in range(300):
    r = extract_robot(obs)
    dx = np.clip((target_x - r['x']), -0.05, 0.05)
    dy = np.clip((target_y - r['y']), -0.05, 0.05)
    dtheta = np.clip(((0.0 - r['theta'] + np.pi) % (2*np.pi) - np.pi)*2.0, -0.196, 0.196)
    obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, -0.10, 0], dtype=np.float32))
    if abs(r['x']-target_x)<0.005 and abs(r['y']-target_y)<0.005 and abs(r['theta'])<0.05:
        break

r = extract_robot(obs)
print(f"\nAt: ({r['x']:.4f},{r['y']:.4f}) theta={r['theta']:.4f} arm={r['arm_joint']:.4f}")

# Now try arm extension alone (no dx)
print("\nArm extension alone (no movement):")
for step in range(5):
    r = extract_robot(obs)
    v = 1.0 if r['arm_joint'] >= 0.16 else 0.0
    obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0.10, v], dtype=np.float32))
    r2 = extract_robot(obs)
    print(f"  step={step}: arm {r['arm_joint']:.4f}->{r2['arm_joint']:.4f} vac={r2['vacuum']:.0f}")
    if r2['arm_joint'] > 0.15:
        print(f"  ARM EXTENDED to {r2['arm_joint']:.4f}!")
        break

r = extract_robot(obs)
ob2 = extract_obstruction(obs, 0)
print(f"\nAfter arm ext: robot=({r['x']:.4f},{r['y']:.4f}) arm={r['arm_joint']:.4f} vac={r['vacuum']:.0f}")
print(f"Obs0 cx: {ob2['cx']:.4f} (moved from {ob['cx']:.4f})")
