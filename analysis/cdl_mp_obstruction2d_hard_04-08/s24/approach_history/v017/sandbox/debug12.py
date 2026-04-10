import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_obstruction, IDX_OBS0_BASE, OBS_STRIDE

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)

# Print all obstruction raw data
for i in range(4):
    b = IDX_OBS0_BASE + i * OBS_STRIDE
    print(f"Obs{i} raw obs[{b}:{b+10}]: {obs[b:b+10]}")
    ob = extract_obstruction(obs, i)
    print(f"  static={ob['static']:.3f}, cx={ob['cx']:.4f}, cy={ob['cy']:.4f}, y2={ob['y2']:.4f}")

# From robot at (0.5484, 0.4319) theta=-pi/2, arm=0.20:
# Suction center = (0.5484, 0.4319 - 0.215) = (0.5484, 0.2169)
# For obs0: x=[0.4912, 0.6056], y=[0.10, 0.2219]
# Suction center inside? x: 0.4912<=0.5484<=0.6056 YES, y: 0.10<=0.2169<=0.2219 YES
print("\nManual test: extend arm to 0.20 at above position, then check vacuum")

ob0 = extract_obstruction(obs, 0)
# Navigate to exact position
target_x = ob0['cx']
target_y = ob0['y2'] + 0.21  # = 0.4319
print(f"Target: ({target_x:.4f},{target_y:.4f})")

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
print(f"At: ({r['x']:.4f},{r['y']:.4f}) theta={r['theta']:.4f} arm={r['arm_joint']:.4f}")

# Now extend arm very slowly (small darm steps)
print("\nExtending arm slowly:")
for step in range(30):
    r = extract_robot(obs)
    v = 1.0 if r['arm_joint'] >= 0.16 else 0.0
    obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0.02, v], dtype=np.float32))
    r2 = extract_robot(obs)
    ob2 = extract_obstruction(obs, 0)
    moved = abs(ob2['cx'] - ob0['cx']) > 0.001
    if r2['arm_joint'] != r['arm_joint'] or step < 5:
        print(f"  step={step}: arm {r['arm_joint']:.4f}->{r2['arm_joint']:.4f} vac={r2['vacuum']:.0f} obs_cx={ob2['cx']:.4f} MOVED={moved}")
    if moved:
        print("  *** GRASP! ***")
        break
