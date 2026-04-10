import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_obstruction
from act_helpers import GRASP_REACH

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
ob = extract_obstruction(obs, 0)
x1 = ob['x1']
print(f"Obs0: x1={x1:.4f} x2={ob['x2']:.4f} y1={ob['y1']:.4f} y2={ob['y2']:.4f}")
print(f"GRASP_REACH = {GRASP_REACH:.4f}")

grip_w = 0.01
arm_max = 0.20

for target_x_offset, label in [
    (GRASP_REACH, "obj_x1 - GRASP_REACH (suction at face)"),
    (GRASP_REACH - 0.005, "obj_x1 - GRASP_REACH + 0.005"),
    (GRASP_REACH - 0.01, "obj_x1 - GRASP_REACH + 0.010"),
    (0.205, "obj_x1 - 0.205 (old formula)"),
]:
    rx = x1 - target_x_offset
    arm_contact = x1 - rx - grip_w  # gripper front touches x1
    full_ext_pen = rx + arm_max + grip_w - x1  # penetration at full ext
    suction_at_max = rx + arm_max + 1.5*grip_w
    print(f"\n{label}: robot_x={rx:.4f}")
    print(f"  arm_joint at contact: {arm_contact:.4f} (need >= 0.16)")
    print(f"  full_ext penetration: {full_ext_pen:.4f} (need <= ~0.005)")
    print(f"  suction at full ext: {suction_at_max:.4f} vs x1={x1:.4f} (inside? {suction_at_max > x1})")

def test_grasp(target_x, label):
    env.reset(seed=0)
    obs2, _ = env.reset(seed=0)
    ob2 = extract_obstruction(obs2, 0)
    target_y = max(0.22, ob2['cy'])
    
    # Navigate
    for step in range(300):
        r = extract_robot(obs2)
        dx = np.clip((target_x - r['x']), -0.05, 0.05)
        dy = np.clip((target_y - r['y']), -0.05, 0.05)
        dtheta = np.clip(((0.0 - r['theta'] + np.pi) % (2*np.pi) - np.pi)*2.0, -0.196, 0.196)
        obs2, _, _, _, _ = env.step(np.array([dx, dy, dtheta, -0.10, 0], dtype=np.float32))
        if abs(r['x']-target_x)<0.005 and abs(r['y']-target_y)<0.005 and abs(r['theta'])<0.05:
            break
    
    r = extract_robot(obs2)
    print(f"\nTesting {label} from ({r['x']:.4f},{r['y']:.4f}):")
    
    # Extend arm with vacuum
    initial_cx = extract_obstruction(obs2, 0)['cx']
    for step in range(5):
        r = extract_robot(obs2)
        obs2, _, _, _, _ = env.step(np.array([0, 0, 0, 0.10, 1.0], dtype=np.float32))
        r2 = extract_robot(obs2)
        ob2 = extract_obstruction(obs2, 0)
        moved = ob2['cx'] != initial_cx
        print(f"  step={step}: arm {r['arm_joint']:.3f}->{r2['arm_joint']:.3f} vac={r2['vacuum']:.0f} obs_cx={ob2['cx']:.4f} MOVED={moved}")
        if moved:
            print("  *** GRASP SUCCEEDED! ***")
            break

test_grasp(x1 - GRASP_REACH, "GRASP_REACH")
test_grasp(x1 - GRASP_REACH + 0.005, "GRASP_REACH-0.005")
test_grasp(x1 - 0.205, "old formula")
