import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_obstruction

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
ob0 = extract_obstruction(obs, 0)
target_x = ob0['cx']
target_y = ob0['y2'] + 0.21

for step in range(300):
    r = extract_robot(obs)
    dx = np.clip((target_x - r['x']), -0.05, 0.05)
    dy = np.clip((target_y - r['y']), -0.05, 0.05)
    dtheta = np.clip(((-np.pi/2 - r['theta'] + np.pi) % (2*np.pi) - np.pi)*2.0, -0.196, 0.196)
    obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, -0.10, 0], dtype=np.float32))
    r2 = extract_robot(obs)
    if abs(r2['x']-target_x)<0.005 and abs(r2['y']-target_y)<0.005 and abs(r2['arm_joint']-0.10)<0.005:
        break

# Extend with vacuum on from start (to cover the "one more step" case)
print("Extending with v=1.0 always:")
for step in range(10):
    r = extract_robot(obs)
    obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0.10, 1.0], dtype=np.float32))
    r2 = extract_robot(obs)
    ob2 = extract_obstruction(obs, 0)
    moved = abs(ob2['cx'] - ob0['cx']) > 0.001
    print(f"  step={step}: arm {r['arm_joint']:.4f}->{r2['arm_joint']:.4f} vac={r2['vacuum']:.0f} obs_cx={ob2['cx']:.4f} MOVED={moved}")
    if moved:
        print(f"  *** GRASPED! Moving with robot...")
        for s in range(8):
            obs, _, _, _, _ = env.step(np.array([-0.04, 0.04, 0, -0.10, 1.0], dtype=np.float32))
            ob3 = extract_obstruction(obs, 0)
            r3 = extract_robot(obs)
            print(f"    s={s}: robot=({r3['x']:.3f},{r3['y']:.3f}) obs_cx={ob3['cx']:.4f}")
        break
