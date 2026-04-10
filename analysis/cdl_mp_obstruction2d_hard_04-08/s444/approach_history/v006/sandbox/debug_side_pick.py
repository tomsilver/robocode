"""Test: approach obs0 from SIDE (theta=0, arm right) and pick with horizontal arm."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, obstruction_overlaps_surface, SUCTION_EXTRA, OBJ_TOP_OFFSET, PICK_ARM_JOINT
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA, K_POS, MAX_DX, MAX_DY, NAV_HIGH_Y
import numpy as np

# Strategy: approach obs0 from LEFT side with arm horizontal (theta=0)
# Robot at x = obs0_left - robot_rad - small_margin, y = obs0_center_y
# Arm extends RIGHT to obs0 center; suction enters obs0

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)
o0 = get_obstruction(obs, 0)
print(f"obs0: x={o0['x']:.3f} y={o0['y']:.4f} w={o0['width']:.4f} h={o0['height']:.4f}")

# obs0 parameters
obs0_x = o0['x']  # 0.491
obs0_y = o0['y']  # 0.100 (bottom? center?)
obs0_h = o0['height']  # 0.122

# For horizontal arm (theta=0): arm extends right
# suction at arm_tip + 0.015 in +x direction
# Approach from left: robot_x + arm = obs0_x (arm tip at obs0 center_x)
# Robot_y = obs0_y (try both center and bottom interpretations)

for interp, robot_y_target in [('center_y_interpretation', obs0_y + obs0_h/2), 
                                  ('bottom_y_interpretation', obs0_y + obs0_h/2)]:
    # robot needs arm_joint such that suction is inside obs0
    # With theta=0: suction_x = robot_x + arm_joint + 0.015
    # For suction_x to be at obs0_x: arm_joint = obs0_x - 0.015 - robot_x
    # Robot at x = obs0_x - arm_min_approach
    arm_target = PICK_ARM_JOINT  # 0.13
    robot_x_target = obs0_x - arm_target - 0.015 - 0.005  # suction just inside obs0 left edge
    print(f"\n{interp}: robot_x={robot_x_target:.3f} y={robot_y_target:.3f} arm={arm_target:.3f}")
    print(f"  suction_x would be: {robot_x_target + arm_target + 0.015:.4f} obs0_x={obs0_x:.4f}")

# Just try direct: low y (min possible), arm=0.10, vacuum to see if it picks anything
print("\n=== Test: low y (y=0.25), arm=0.10, vacuum at obs0 ===")
env2 = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs2, info2 = env2.reset(seed=0)
for step in range(200):
    r = get_robot(obs2)
    rx, ry, theta, arm = r['x'], r['y'], r['theta'], r['arm_joint']
    err = normalize_angle(-np.pi/2 - theta)
    dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
    dx = np.clip((obs0_x - rx) * K_POS, -MAX_DX, MAX_DX)
    if step < 80:
        dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY); vac=0
    else:
        dy = np.clip((0.25 - ry) * K_POS, -MAX_DY, MAX_DY); vac=1.0 if step>120 else 0
    action = np.array([dx, dy, dtheta, 0, vac], dtype=np.float32)
    obs2, _, _, _, _ = env2.step(action)
    
    if not obstruction_overlaps_surface(obs2, 0):
        print(f"  CLEARED at step {step}!")
        break
    if step in [119, 199]:
        r2 = get_robot(obs2)
        o = get_obstruction(obs2, 0)
        suc_y = r2['y'] + r2['arm_joint']*np.sin(r2['theta']) + 0.015*np.sin(r2['theta'])
        print(f"  step={step} robot=({r2['x']:.3f},{r2['y']:.4f}) arm={r2['arm_joint']:.4f} theta={r2['theta']:.3f} suc_y={suc_y:.4f} obs0_y={o['y']:.4f}")
