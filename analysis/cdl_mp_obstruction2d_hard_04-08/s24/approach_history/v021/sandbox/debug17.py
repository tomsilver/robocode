import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_block, extract_obstruction, gripper_tip_pos

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
blk = extract_block(obs)
print(f"Block: x=[{blk['x1']:.4f},{blk['x2']:.4f}] y=[{blk['y1']:.4f},{blk['y2']:.4f}] cx={blk['cx']:.4f} cy={blk['cy']:.4f}")

# Try above approach on BLOCK
target_x = blk['cx']
target_y = blk['y2'] + 0.21
print(f"Target: ({target_x:.4f},{target_y:.4f}), suc_y={target_y-0.215:.4f} vs blk_y2={blk['y2']:.4f}")

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
print(f"\nAt: ({r['x']:.4f},{r['y']:.4f}) theta={r['theta']:.4f} arm={r['arm_joint']:.4f}")

# Extend with v=1
for step in range(10):
    r = extract_robot(obs)
    obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0.10, 1.0], dtype=np.float32))
    r2 = extract_robot(obs)
    blk2 = extract_block(obs)
    moved = abs(blk2['cy'] - blk['cy']) > 0.001
    tip = gripper_tip_pos(obs)
    print(f"  step={step}: arm {r['arm_joint']:.4f}->{r2['arm_joint']:.4f} vac={r2['vacuum']:.0f} blk_cy={blk2['cy']:.4f} MOVED={moved} suc=({tip[0]:.4f},{tip[1]:.4f})")
    if moved:
        print("  *** BLOCK GRASPED! ***")
        break

# Also test grasping an obstruction using the exact same gripper_tip_pos check
print("\n--- Now checking: does obstruction have a different 'static' value?")
for i in range(4):
    ob = extract_obstruction(obs, i)
    print(f"Obs{i}: static={ob['static']:.3f}, y2={ob['y2']:.4f}, cx={ob['cx']:.4f}")
print(f"Block: check obs[19:29]")
print(f"Block raw: {obs[19:29]}")
