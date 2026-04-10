import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_block

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
blk0 = extract_block(obs)
print(f"Block: cx={blk0['cx']:.4f} cy={blk0['cy']:.4f} y2={blk0['y2']:.4f}")

# Navigate ABOVE block
target_x = blk0['cx']
target_y = blk0['y2'] + 0.21

for step in range(300):
    r = extract_robot(obs)
    dx = np.clip((target_x - r['x']), -0.05, 0.05)
    dy = np.clip((target_y - r['y']), -0.05, 0.05)
    dtheta = np.clip(((-np.pi/2 - r['theta'] + np.pi) % (2*np.pi) - np.pi)*2.0, -0.196, 0.196)
    obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, -0.10, 0], dtype=np.float32))
    r2 = extract_robot(obs)
    if abs(r2['x']-target_x)<0.005 and abs(r2['y']-target_y)<0.005 and abs(r2['arm_joint']-0.10)<0.005:
        break

# Extend arm with vac=1, then move robot and watch block
print("Extending and moving robot:")
for step in range(20):
    r = extract_robot(obs)
    blk = extract_block(obs)
    # Move robot left while keeping vacuum on
    if step < 3:
        action = np.array([0, 0, 0, 0.10, 1.0], dtype=np.float32)  # just extend
    else:
        action = np.array([-0.03, 0, 0, 0, 1.0], dtype=np.float32)  # move left
    obs, _, _, _, _ = env.step(action)
    r2 = extract_robot(obs)
    blk2 = extract_block(obs)
    print(f"  step={step}: robot_x={r2['x']:.4f} arm={r2['arm_joint']:.4f} vac={r2['vacuum']:.0f} blk_cx={blk2['cx']:.4f}")
    if abs(blk2['cx'] - blk0['cx']) > 0.005:
        print("  *** Block moved! Held! ***")
        break
