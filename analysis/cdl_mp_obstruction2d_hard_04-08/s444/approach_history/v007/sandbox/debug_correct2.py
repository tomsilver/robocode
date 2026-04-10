"""Test: descend with arm RETRACTED, then extend to 0.163, vacuum, lift."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, obstruction_overlaps_surface, NUM_OBSTRUCTIONS
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA, K_POS, MAX_DX, MAX_DY, NAV_HIGH_Y
import numpy as np

ROB_RADIUS = 0.10
GRIPPER_HEIGHT = 0.07
MARGIN = ROB_RADIUS + GRIPPER_HEIGHT   # = 0.17
OBS_PICK_ARM = 0.163  # arm_joint for picking (just enough for suction inside obs)

def pick_obs(seed, obs_idx):
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=seed)
    o = get_obstruction(obs, obs_idx)
    obs_x = o['x']
    obs_top = o['y'] + o['height'] / 2   # center formula
    pick_y = obs_top + MARGIN             # = obs_top + 0.17
    
    print(f"  Seed {seed} obs{obs_idx}: x={obs_x:.3f} obs_top={obs_top:.3f} pick_y={pick_y:.3f} arm={OBS_PICK_ARM:.3f}")
    print(f"  suction_y will be: {pick_y - OBS_PICK_ARM - 0.015:.4f} (want < obs_top={obs_top:.4f})")

    for step in range(400):
        r = get_robot(obs)
        rx, ry, theta, arm = r['x'], r['y'], r['theta'], r['arm_joint']
        err = normalize_angle(-np.pi/2 - theta)
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
        dx = np.clip((obs_x - rx) * K_POS, -MAX_DX, MAX_DX)

        if step < 80:  # Nav to obs_x, high y, ARM RETRACTED
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((0.10 - arm) * 8, -0.1, 0.1); vac = 0
        elif step < 160:  # Descend to pick_y with ARM RETRACTED
            dy = np.clip((pick_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((0.10 - arm) * 8, -0.1, 0.1); vac = 0  # keep retracted!
        elif step < 220:  # NOW extend arm to 0.163
            dy = np.clip((pick_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((OBS_PICK_ARM - arm) * 8, -0.1, 0.1); vac = 0
        elif step < 260:  # Vacuum on
            dy = np.clip((pick_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((OBS_PICK_ARM - arm) * 8, -0.1, 0.1); vac = 1.0
        else:  # Rise to NAV_HIGH_Y, arm retract
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((0.10 - arm) * 8, -0.1, 0.1); vac = 1.0

        action = np.array([dx, dy, dtheta, darm, vac], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)

        if not obstruction_overlaps_surface(obs, obs_idx):
            r2 = get_robot(obs)
            o2 = get_obstruction(obs, obs_idx)
            print(f"  CLEARED at step {step}! robot=({r2['x']:.3f},{r2['y']:.3f}) obs=({o2['x']:.3f},{o2['y']:.3f})")
            return True

        if step in [159, 219, 259, 299, 399]:
            r2 = get_robot(obs)
            o2 = get_obstruction(obs, obs_idx)
            suc_y = r2['y'] - r2['arm_joint'] - 0.015
            print(f"  step={step} robot=({r2['x']:.3f},{r2['y']:.4f}) arm={r2['arm_joint']:.4f} suc_y={suc_y:.4f} obs_y={o2['y']:.4f}")
        if terminated or truncated: break
    return False

for seed in [0, 1, 2]:
    print(f"\nSeed {seed}:")
    for i in range(NUM_OBSTRUCTIONS):
        env2 = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
        obs2, _ = env2.reset(seed=seed)
        if obstruction_overlaps_surface(obs2, i):
            pick_obs(seed, i)
