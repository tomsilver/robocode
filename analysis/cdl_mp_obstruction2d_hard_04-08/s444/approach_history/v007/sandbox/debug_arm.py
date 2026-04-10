"""Debug: pick obstruction from stuck_y by extending arm carefully."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_obstruction, get_robot
import numpy as np

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

# Compute geometry
o0_y, o0_h = 0.100, 0.122
o0_top = o0_y + o0_h/2  # 0.161
stuck_y = 0.331  # empirical min_y above obs0
SUCTION_EXTRA = 0.015
# gripper extends ±width/2 along arm, ±height/2 perpendicular
# With arm down (theta=-pi/2): gripper_width=0.01 along arm (vertical), height=0.07 perpendicular
GRIPPER_ALONG_ARM = 0.01   # dimension along arm direction (vertical when arm down)
GRIPPER_PERP = 0.07        # dimension perpendicular to arm (horizontal when arm down)

print(f"obs0 top: {o0_top:.4f}")
print(f"stuck_y: {stuck_y:.4f}")
for arm in np.arange(0.10, 0.21, 0.005):
    arm_tip_y = stuck_y - arm  # below robot
    gripper_center_y = arm_tip_y
    gripper_bottom_y = gripper_center_y - GRIPPER_ALONG_ARM/2  # farthest down
    suction_y = arm_tip_y - SUCTION_EXTRA
    gripper_hits = gripper_bottom_y <= o0_top
    suction_in = suction_y <= o0_top
    print(f"  arm={arm:.3f}: tip={arm_tip_y:.4f} grip_bot={gripper_bottom_y:.4f} "
          f"suc_y={suction_y:.4f} | grip_hits={gripper_hits} suc_in={suction_in}")

# Now simulate: navigate to (obs0_x, stuck_y), extend arm to 0.16, apply vacuum
from act_helpers import clip_action, normalize_angle, K_THETA, K_POS, MAX_DX, MAX_DY, MAX_DTHETA
obs0_x = 0.491
target_arm = 0.16
NAV_HIGH_Y = 0.50

print("\n--- Navigate to above obs0 ---")
# Phase 1: nav high
for step in range(200):
    r = get_robot(obs)
    rx, ry = r['x'], r['y']
    theta = r['theta']
    arm = r['arm_joint']

    err = normalize_angle(-np.pi/2 - theta)
    dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)

    if step < 60:
        # Navigate to (obs0_x, NAV_HIGH_Y)
        dx = np.clip((obs0_x - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
        action = np.array([dx, dy, dtheta, 0.0, 0.0], dtype=np.float32)
    elif step < 120:
        # Descend to stuck_y (will stop at min_y)
        dx = np.clip((obs0_x - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((stuck_y - ry) * K_POS, -MAX_DY, MAX_DY)
        action = np.array([dx, dy, dtheta, 0.0, 0.0], dtype=np.float32)
    elif step < 160:
        # Extend arm to target_arm
        dx = np.clip((obs0_x - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((stuck_y - ry) * K_POS, -MAX_DY, MAX_DY)
        darm = np.clip((target_arm - arm) * 8.0, -0.1, 0.1)
        action = np.array([dx, dy, dtheta, darm, 0.0], dtype=np.float32)
    else:
        # Apply vacuum
        dx = np.clip((obs0_x - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((stuck_y - ry) * K_POS, -MAX_DY, MAX_DY)
        darm = np.clip((target_arm - arm) * 8.0, -0.1, 0.1)
        action = np.array([dx, dy, dtheta, darm, 1.0], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)
    r2 = get_robot(obs)
    o0 = get_obstruction(obs, 0)

    if step % 20 == 0 or step >= 160:
        print(f"step={step:3d}: robot=({r2['x']:.3f},{r2['y']:.3f}) "
              f"arm={r2['arm_joint']:.4f} theta={r2['theta']:.3f} "
              f"obs0=({o0['x']:.3f},{o0['y']:.3f})")

    if terminated or truncated:
        print(f"Done! reward={reward}")
        break
