import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_obstruction, gripper_tip_pos

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
ob0 = extract_obstruction(obs, 0)
print(f"Obs0 raw: y1={ob0['y1']:.6f} y2={ob0['y2']:.6f} height={ob0['height']:.6f}")

target_y = ob0['y2'] + 0.21
for step in range(300):
    r = extract_robot(obs)
    dx = np.clip((ob0['cx'] - r['x']), -0.05, 0.05)
    dy = np.clip((target_y - r['y']), -0.05, 0.05)
    dtheta = np.clip(((-np.pi/2 - r['theta'] + np.pi) % (2*np.pi) - np.pi)*2.0, -0.196, 0.196)
    obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, -0.10, 0], dtype=np.float32))
    r2 = extract_robot(obs)
    if abs(r2['x']-ob0['cx'])<0.003 and abs(r2['y']-target_y)<0.003 and abs(r2['arm_joint']-0.10)<0.003:
        break

r = extract_robot(obs)
print(f"\nRobot at: ({r['x']:.6f},{r['y']:.6f}) theta={r['theta']:.6f} arm={r['arm_joint']:.6f}")
tip = gripper_tip_pos(obs)
print(f"Suction center (arm_min): {tip}")

# Extend arm to max (one big step with v=1)
obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0.10, 1.0], dtype=np.float32))
r2 = extract_robot(obs)
tip2 = gripper_tip_pos(obs)
ob_after = extract_obstruction(obs, 0)
print(f"\nAfter extend: robot_y={r2['y']:.6f} arm={r2['arm_joint']:.6f} vac={r2['vacuum']:.0f}")
print(f"Suction center (arm_max): ({tip2[0]:.6f}, {tip2[1]:.6f})")
print(f"Obs0 y2={ob_after['y2']:.6f}, suction_y - y1={tip2[1]-ob_after['y1']:.6f}, y2-suction_y={ob_after['y2']-tip2[1]:.6f}")
print(f"Obs0 x range: [{ob_after['x1']:.6f}, {ob_after['x2']:.6f}], suction_x={tip2[0]:.6f}")
print(f"MOVED: {abs(ob_after['cx']-ob0['cx'])>0.001 or abs(ob_after['cy']-ob0['cy'])>0.001}")

# Check if one more step triggers it
obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0, 1.0], dtype=np.float32))
ob2 = extract_obstruction(obs, 0)
print(f"\nAfter holding vacuum 1 more step: MOVED={abs(ob2['cy']-ob0['cy'])>0.001}")
print(f"obs_cy: before={ob0['cy']:.6f} after={ob2['cy']:.6f}")
