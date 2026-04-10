import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_obstruction
from act_helpers import GRASP_REACH

GRIP_W = 0.01

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
ob = extract_obstruction(obs, 0)
print(f"Obs0: x1={ob['x1']:.4f} y2={ob['y2']:.4f} cx={ob['cx']:.4f}")

# Above approach: robot at (obj_cx, obj_y2 + GRASP_REACH - GRIP_W) with theta=-pi/2
target_x = ob['cx']
target_y = ob['y2'] + GRASP_REACH - GRIP_W  # suction at obj_y2 + GRIP_W - GRIP_W ... 
# Actually: suction at robot_y - arm_max - 1.5*GRIP_W
# Want suction at obj_y2 + 0.005 (inside):
# robot_y = obj_y2 + 0.005 + arm_max + 1.5*GRIP_W = obj_y2 + 0.005 + 0.215 = obj_y2 + 0.220
target_y = ob['y2'] + 0.220  # suction 5mm into top face
target_theta = -np.pi / 2

print(f"Target above: ({target_x:.4f},{target_y:.4f}) theta={target_theta:.4f}")
suction_y = target_y - 0.215
print(f"  Suction y = {suction_y:.4f}, obs y2 = {ob['y2']:.4f}, inside = {suction_y < ob['y2']}")

# Navigate there
obs2, _ = env.reset(seed=0)
for step in range(300):
    r = extract_robot(obs2)
    dx = np.clip((target_x - r['x']), -0.05, 0.05)
    dy = np.clip((target_y - r['y']), -0.05, 0.05)
    dtheta = np.clip(((target_theta - r['theta'] + np.pi) % (2*np.pi) - np.pi)*2.0, -0.196, 0.196)
    obs2, _, _, _, _ = env.step(np.array([dx, dy, dtheta, -0.10, 0], dtype=np.float32))
    r2 = extract_robot(obs2)
    if abs(r2['x']-target_x)<0.01 and abs(r2['y']-target_y)<0.01 and abs((r2['theta']-target_theta+np.pi)%(2*np.pi)-np.pi)<0.1:
        break

r = extract_robot(obs2)
print(f"\nAt: ({r['x']:.4f},{r['y']:.4f}) theta={r['theta']:.4f}")

# Extend arm with vacuum
initial_cx = extract_obstruction(obs2, 0)['cx']
for step in range(10):
    r = extract_robot(obs2)
    obs2, _, _, _, _ = env.step(np.array([0, 0, 0, 0.10, 1.0], dtype=np.float32))
    r2 = extract_robot(obs2)
    ob2 = extract_obstruction(obs2, 0)
    moved = abs(ob2['cx'] - initial_cx) > 0.001
    print(f"  step={step}: arm {r['arm_joint']:.3f}->{r2['arm_joint']:.3f} vac={r2['vacuum']:.0f} obs_cx={ob2['cx']:.4f} MOVED={moved}")
    if moved:
        print("  *** GRASP SUCCEEDED! ***")
        break
    if step == 0 and r2['arm_joint'] < 0.15:
        print("  Arm blocked!")

# Try moving robot slightly lower
print("\n--- Trying robot at obj_y2 + GRASP_REACH (suction exactly at face):")
obs3, _ = env.reset(seed=0)
ob3 = extract_obstruction(obs3, 0)
target_y2 = ob3['y2'] + GRASP_REACH  # suction exactly at y2
for step in range(300):
    r = extract_robot(obs3)
    dx = np.clip((ob3['cx'] - r['x']), -0.05, 0.05)
    dy = np.clip((target_y2 - r['y']), -0.05, 0.05)
    dtheta = np.clip(((target_theta - r['theta'] + np.pi) % (2*np.pi) - np.pi)*2.0, -0.196, 0.196)
    obs3, _, _, _, _ = env.step(np.array([dx, dy, dtheta, -0.10, 0], dtype=np.float32))
    r2 = extract_robot(obs3)
    if abs(r2['x']-ob3['cx'])<0.01 and abs(r2['y']-target_y2)<0.01 and abs((r2['theta']-target_theta+np.pi)%(2*np.pi)-np.pi)<0.1:
        break

r = extract_robot(obs3)
print(f"At: ({r['x']:.4f},{r['y']:.4f}) theta={r['theta']:.4f}")
initial_cx3 = extract_obstruction(obs3, 0)['cx']
for step in range(10):
    r = extract_robot(obs3)
    obs3, _, _, _, _ = env.step(np.array([0, 0, 0, 0.10, 1.0], dtype=np.float32))
    r2 = extract_robot(obs3)
    ob3 = extract_obstruction(obs3, 0)
    moved = abs(ob3['cx'] - initial_cx3) > 0.001
    print(f"  step={step}: arm {r['arm_joint']:.3f}->{r2['arm_joint']:.3f} vac={r2['vacuum']:.0f} MOVED={moved}")
    if moved:
        print("  *** GRASP SUCCEEDED! ***")
        break
