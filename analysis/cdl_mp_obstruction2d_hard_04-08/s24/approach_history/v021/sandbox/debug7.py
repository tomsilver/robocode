import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_obstruction

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)

ob = extract_obstruction(obs, 0)
target_x = ob['x1'] - 0.230
target_y = max(0.22, ob['cy'])

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
print(f"At: ({r['x']:.4f},{r['y']:.4f}) theta={r['theta']:.4f} arm={r['arm_joint']:.4f}")

# Extend arm with vacuum ON from start
print("\nArm extension WITH vacuum=1 from start:")
for step in range(10):
    r = extract_robot(obs)
    obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0.10, 1.0], dtype=np.float32))
    r2 = extract_robot(obs)
    ob2 = extract_obstruction(obs, 0)
    print(f"  step={step}: arm {r['arm_joint']:.4f}->{r2['arm_joint']:.4f} vac={r2['vacuum']:.0f} obs0_cx={ob2['cx']:.4f}")
    if ob2['cx'] != ob['cx']:
        print("  OBS0 MOVED! Grasp succeeded!")
        break
