import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_obstruction

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
ob = extract_obstruction(obs, 0)
print(f"Obs0: cx={ob['cx']:.4f} y2={ob['y2']:.4f}")

ARM_MAX = 0.20
GRIP_W = 0.01
# above approach: gripper front exactly at obj_y2, suction 5mm inside
target_x = ob['cx']
target_y = ob['y2'] + ARM_MAX + GRIP_W  # = obj_y2 + 0.21
print(f"Target: ({target_x:.4f},{target_y:.4f})")
print(f"  suction_y = {target_y - 0.215:.4f} vs obj_y2={ob['y2']:.4f} (inside={target_y - 0.215 < ob['y2']})")

obs2, _ = env.reset(seed=0)
for step in range(300):
    r = extract_robot(obs2)
    dx = np.clip((target_x - r['x']), -0.05, 0.05)
    dy = np.clip((target_y - r['y']), -0.05, 0.05)
    dtheta = np.clip(((-np.pi/2 - r['theta'] + np.pi) % (2*np.pi) - np.pi)*2.0, -0.196, 0.196)
    obs2, _, _, _, _ = env.step(np.array([dx, dy, dtheta, -0.10, 0], dtype=np.float32))
    r2 = extract_robot(obs2)
    if abs(r2['x']-target_x)<0.01 and abs(r2['y']-target_y)<0.01 and abs((r2['theta']+np.pi/2+np.pi)%(2*np.pi)-np.pi)<0.1:
        break

r = extract_robot(obs2)
print(f"At: ({r['x']:.4f},{r['y']:.4f}) theta={r['theta']:.4f}")
initial_cx = extract_obstruction(obs2, 0)['cx']

for step in range(10):
    r = extract_robot(obs2)
    obs2, _, _, _, _ = env.step(np.array([0, 0, 0, 0.10, 1.0], dtype=np.float32))
    r2 = extract_robot(obs2)
    ob2 = extract_obstruction(obs2, 0)
    moved = abs(ob2['cx'] - initial_cx) > 0.001
    print(f"  step={step}: arm {r['arm_joint']:.4f}->{r2['arm_joint']:.4f} vac={r2['vacuum']:.0f} obs_cx={ob2['cx']:.4f} MOVED={moved}")
    if moved:
        print("  *** GRASP SUCCEEDED! ***")
        # Now verify the obstruction moves WITH the robot
        for s in range(5):
            obs2, _, _, _, _ = env.step(np.array([-0.05, 0, 0, -0.10, 1.0], dtype=np.float32))
            ob3 = extract_obstruction(obs2, 0)
            print(f"    move step {s}: obs_cx={ob3['cx']:.4f}")
        break
