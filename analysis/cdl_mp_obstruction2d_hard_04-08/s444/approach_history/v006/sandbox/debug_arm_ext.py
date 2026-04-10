"""Test arm extension at OBS_PICK_OFFSET=0.19 (pick_y=obs_top+0.19)."""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, obstruction_overlaps_surface
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA, K_POS, MAX_DX, MAX_DY
import numpy as np

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)
o = get_obstruction(obs, 0)
obs_top = o['y'] + o['height']/2
obs_x = o['x']
OBS_PICK_OFFSET = 0.19
OBS_PICK_ARM = 0.18  # test with 0.18
pick_y = obs_top + OBS_PICK_OFFSET
print(f"obs_top={obs_top:.4f} pick_y={pick_y:.4f} arm={OBS_PICK_ARM}")
print(f"Expected: arm_tip_y={pick_y-OBS_PICK_ARM:.4f} (should be > {obs_top:.4f})")
print(f"Expected: suction_y={pick_y-OBS_PICK_ARM-0.015:.4f} (should be < {obs_top:.4f})")

for step in range(400):
    r = get_robot(obs)
    rx, ry, theta, arm = r['x'], r['y'], r['theta'], r['arm_joint']
    err = normalize_angle(-np.pi/2 - theta)
    dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)

    if step < 80:  # nav high
        dx = np.clip((obs_x - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((0.50 - ry) * K_POS, -MAX_DY, MAX_DY)
        darm, vac = 0, 0
    elif step < 160:  # descend to pick_y (arm retracted)
        dx = np.clip((obs_x - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((pick_y - ry) * K_POS, -MAX_DY, MAX_DY)
        darm, vac = 0, 0
    elif step < 240:  # extend arm
        dx = np.clip((obs_x - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((pick_y - ry) * K_POS, -MAX_DY, MAX_DY)
        darm = np.clip((OBS_PICK_ARM - arm) * 8.0, -0.1, 0.1)
        vac = 0
    elif step < 280:  # vacuum on
        dx = np.clip((obs_x - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((pick_y - ry) * K_POS, -MAX_DY, MAX_DY)
        darm = np.clip((OBS_PICK_ARM - arm) * 8.0, -0.1, 0.1)
        vac = 1.0
    else:  # rise
        dx = np.clip((obs_x - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((0.60 - ry) * K_POS, -MAX_DY, MAX_DY)
        darm = np.clip((0.10 - arm) * 8.0, -0.1, 0.1)
        vac = 1.0

    action = np.array([dx, dy, dtheta, darm, vac], dtype=np.float32)
    obs, _, terminated, truncated, _ = env.step(action)
    r2 = get_robot(obs)

    if step in [79, 159, 160, 180, 200, 239, 240, 260, 280, 300, 350, 399]:
        o0 = get_obstruction(obs, 0)
        arm_tip_y = r2['y'] - r2['arm_joint']  # theta=-pi/2
        suc_y = arm_tip_y - 0.015
        on = obstruction_overlaps_surface(obs, 0)
        print(f"  step={step:3d} robot=({r2['x']:.3f},{r2['y']:.4f}) arm={r2['arm_joint']:.4f} "
              f"arm_tip_y={arm_tip_y:.4f} suc_y={suc_y:.4f} obs_y={o0['y']:.4f} on={on}")

    if not obstruction_overlaps_surface(obs, 0):
        print(f"  CLEARED at step {step}!")
        break
    if terminated or truncated:
        print(f"  TERMINATED/TRUNCATED at step {step}")
        break
