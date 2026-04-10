"""Test lateral approach to obs0 at y=0.376."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_obstruction, get_robot, obstruction_overlaps_surface
from act_helpers import clip_action, normalize_angle, K_THETA, K_POS, MAX_DX, MAX_DY, MAX_DTHETA, NAV_HIGH_Y
import numpy as np

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

obs0_x = 0.491
obs0_top = 0.161
ARM_MAX = 0.20
SUCTION_EXTRA = 0.015
# We want suction_y = obs_top: robot_y = obs_top + ARM_MAX + SUCTION_EXTRA = 0.376
pick_y = obs0_top + ARM_MAX + SUCTION_EXTRA  # = 0.376
pick_y += 0.01  # small extra margin
approach_x = 0.20  # clear of obs0, descend here
target_arm = ARM_MAX  # = 0.20

print(f"pick_y={pick_y:.4f}, approach_x={approach_x:.3f}, target_arm={target_arm:.3f}")
print(f"Expected suction_y = {pick_y - target_arm - SUCTION_EXTRA:.4f} (want ≤ {obs0_top:.3f})")

for step in range(300):
    r = get_robot(obs)
    rx, ry, theta, arm = r['x'], r['y'], r['theta'], r['arm_joint']

    err = normalize_angle(-np.pi/2 - theta)
    dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)

    if step < 60:
        # Phase 1: nav to clear position (approach_x, NAV_HIGH_Y)
        dx = np.clip((approach_x - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
        action = np.array([dx, dy, dtheta, 0, 0], dtype=np.float32)
    elif step < 120:
        # Phase 2: descend to pick_y at approach_x
        dx = np.clip((approach_x - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((pick_y - ry) * K_POS, -MAX_DY, MAX_DY)
        action = np.array([dx, dy, dtheta, 0, 0], dtype=np.float32)
    elif step < 180:
        # Phase 3: navigate LATERALLY to obs0_x at pick_y
        dx = np.clip((obs0_x - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((pick_y - ry) * K_POS, -MAX_DY, MAX_DY)
        action = np.array([dx, dy, dtheta, 0, 0], dtype=np.float32)
    elif step < 220:
        # Phase 4: extend arm
        dx = np.clip((obs0_x - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((pick_y - ry) * K_POS, -MAX_DY, MAX_DY)
        darm = np.clip((target_arm - arm) * 8.0, -0.1, 0.1)
        action = np.array([dx, dy, dtheta, darm, 0], dtype=np.float32)
    else:
        # Phase 5: apply vacuum
        dx = np.clip((obs0_x - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((pick_y - ry) * K_POS, -MAX_DY, MAX_DY)
        darm = np.clip((target_arm - arm) * 8.0, -0.1, 0.1)
        action = np.array([dx, dy, dtheta, darm, 1.0], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)
    r2 = get_robot(obs)
    o0 = get_obstruction(obs, 0)

    phase = ['nav_high','nav_high','nav_low','lateral','extend','vacuum'][min(step//60, 5)]
    if step % 20 == 0:
        suc_y = r2['y'] + r2['arm_joint'] * np.sin(r2['theta']) - SUCTION_EXTRA
        print(f"step={step:3d} ph={phase} robot=({r2['x']:.3f},{r2['y']:.3f}) "
              f"arm={r2['arm_joint']:.4f} theta={r2['theta']:.3f} "
              f"suc_y={suc_y:.4f} obs0=({o0['x']:.3f},{o0['y']:.3f}) on={obstruction_overlaps_surface(obs,0)}")

    if not obstruction_overlaps_surface(obs, 0):
        print(f"OBS0 CLEARED at step {step}!")
        break
