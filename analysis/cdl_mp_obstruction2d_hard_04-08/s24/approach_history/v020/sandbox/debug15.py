import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_obstruction, gripper_tip_pos

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
ob0 = extract_obstruction(obs, 0)
print(f"Obs0: x=[{ob0['x1']:.4f},{ob0['x2']:.4f}] y=[{ob0['y1']:.4f},{ob0['y2']:.4f}] cx={ob0['cx']:.4f}")

# Test multiple robot_y values to find which one allows grasp
for ry_offset in [0.21, 0.20, 0.19, 0.18, 0.17]:
    target_y = ob0['y2'] + ry_offset
    obs2, _ = env.reset(seed=0)
    ob02 = extract_obstruction(obs2, 0)
    
    for step in range(300):
        r = extract_robot(obs2)
        dx = np.clip((ob02['cx'] - r['x']), -0.05, 0.05)
        dy = np.clip((target_y - r['y']), -0.05, 0.05)
        dtheta = np.clip(((-np.pi/2 - r['theta'] + np.pi) % (2*np.pi) - np.pi)*2.0, -0.196, 0.196)
        obs2, _, _, _, _ = env.step(np.array([dx, dy, dtheta, -0.10, 0], dtype=np.float32))
        r2 = extract_robot(obs2)
        if abs(r2['x']-ob02['cx'])<0.005 and abs(r2['y']-target_y)<0.005:
            break
    
    r = extract_robot(obs2)
    suc_y = r['y'] + np.sin(r['theta']) * (r['arm_joint'] + 0.015)
    
    # Extend arm with v=1
    grasped = False
    for step in range(5):
        r = extract_robot(obs2)
        obs2, _, _, _, _ = env.step(np.array([0, 0, 0, 0.10, 1.0], dtype=np.float32))
        r2 = extract_robot(obs2)
        ob_after = extract_obstruction(obs2, 0)
        if abs(ob_after['cx'] - ob02['cx']) > 0.001 or abs(ob_after['cy'] - ob02['cy']) > 0.001:
            grasped = True
            break
    
    suc_at_max = r['y'] + np.sin(r['theta']) * 0.215
    print(f"ry_offset={ry_offset:.2f}: robot_y={r['y']:.4f} arm={r2['arm_joint']:.4f} suc_y_at_max={suc_at_max:.4f} obs_y2={ob02['y2']:.4f} inside={ob02['y1']<suc_at_max<ob02['y2']} GRASPED={grasped}")
