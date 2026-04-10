import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_obstruction
import numpy as np

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)

# Manually navigate to (0.311, 0.228) with theta=0
# Just send dx=-0.479, dy=-0.632 (from 0.790,0.860 to 0.311,0.228)
r = extract_robot(obs)
print(f"Start: ({r['x']:.4f},{r['y']:.4f}) theta={r['theta']:.4f}")
for step in range(200):
    r = extract_robot(obs)
    dx = (0.3106 - r['x'])
    dy = (0.2281 - r['y'])
    dtheta = ((0.0 - r['theta'] + np.pi) % (2*np.pi) - np.pi) * 2.0
    darm = (0.10 - r['arm_joint'])
    action = np.array([np.clip(dx, -0.05, 0.05), np.clip(dy, -0.05, 0.05), 
                        np.clip(dtheta, -0.196, 0.196), np.clip(darm, -0.10, 0.10), 0], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)
    if abs(r['x']-0.3106)<0.01 and abs(r['y']-0.2281)<0.01 and abs(r['theta'])<0.1:
        print(f"Reached at step {step}: ({r['x']:.4f},{r['y']:.4f}) theta={r['theta']:.4f}")
        break

r = extract_robot(obs)
print(f"At position ({r['x']:.4f},{r['y']:.4f}) theta={r['theta']:.4f}")

# Now try just moving left (dx=-0.05)
print("\nTrying to move LEFT only (no arm extension):")
for step in range(10):
    r = extract_robot(obs)
    obs, _, _, _, _ = env.step(np.array([-0.05, 0, 0, 0, 0], dtype=np.float32))
    r2 = extract_robot(obs)
    print(f"  step={step} x: {r['x']:.4f} -> {r2['x']:.4f} delta={r2['x']-r['x']:.4f}")

r = extract_robot(obs)
print(f"\nAfter moving left: ({r['x']:.4f},{r['y']:.4f})")
print("\nNow try arm extension only:")
for step in range(10):
    r = extract_robot(obs)
    obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0.1, 0], dtype=np.float32))
    r2 = extract_robot(obs)
    print(f"  step={step} arm: {r['arm_joint']:.4f} -> {r2['arm_joint']:.4f}")
