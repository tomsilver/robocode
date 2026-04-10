import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_obstruction

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)

# Navigate to (0.311, 0.228) theta=0
for step in range(200):
    r = extract_robot(obs)
    if abs(r['x']-0.3106)<0.01 and abs(r['y']-0.2281)<0.01 and abs(r['theta'])<0.1:
        break
    dx = np.clip(0.3106 - r['x'], -0.05, 0.05)
    dy = np.clip(0.2281 - r['y'], -0.05, 0.05)
    dtheta = np.clip(((0.0 - r['theta'] + np.pi) % (2*np.pi) - np.pi)*2.0, -0.196, 0.196)
    obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, -0.1, 0], dtype=np.float32))

r = extract_robot(obs)
ob = extract_obstruction(obs, 0)
print(f"At position ({r['x']:.4f},{r['y']:.4f}) theta={r['theta']:.4f} arm={r['arm_joint']:.4f}")
print(f"Obs0: x1={ob['x1']:.4f} x2={ob['x2']:.4f} y1={ob['y1']:.4f} y2={ob['y2']:.4f}")

# Try combined movement + arm extension
print("\nTrying dx=-0.05 + darm=0.10 simultaneously:")
r_before = extract_robot(obs)
obs, _, _, _, _ = env.step(np.array([-0.05, 0, 0, 0.10, 0], dtype=np.float32))
r_after = extract_robot(obs)
print(f"  Robot x: {r_before['x']:.4f} -> {r_after['x']:.4f} (delta={r_after['x']-r_before['x']:.4f})")
print(f"  Arm: {r_before['arm_joint']:.4f} -> {r_after['arm_joint']:.4f} (delta={r_after['arm_joint']-r_before['arm_joint']:.4f})")

# Try small arm extension (step by step)
print("\nResetting to start position...")
obs, info = env.reset(seed=0)
for step in range(200):
    r = extract_robot(obs)
    if abs(r['x']-0.3106)<0.01 and abs(r['y']-0.2281)<0.01 and abs(r['theta'])<0.1:
        break
    dx = np.clip(0.3106 - r['x'], -0.05, 0.05)
    dy = np.clip(0.2281 - r['y'], -0.05, 0.05)
    dtheta = np.clip(((0.0 - r['theta'] + np.pi) % (2*np.pi) - np.pi)*2.0, -0.196, 0.196)
    obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, -0.1, 0], dtype=np.float32))

print("Arm extension step by step (with dx=0):")
for i in range(15):
    r = extract_robot(obs)
    obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0.10, 0], dtype=np.float32))
    r2 = extract_robot(obs)
    print(f"  step={i} arm: {r['arm_joint']:.4f} -> {r2['arm_joint']:.4f} x: {r['x']:.4f} -> {r2['x']:.4f}")
    if r2['arm_joint'] >= 0.15:
        print("  -> Arm at 0.15+!")
        break
