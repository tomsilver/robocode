"""Test pick with corrected obs_top = y + h (y is bottom coord)."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, obstruction_overlaps_surface, OBJ_TOP_OFFSET, PICK_ARM_JOINT
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA, K_POS, MAX_DX, MAX_DY, NAV_HIGH_Y
import numpy as np

def test_pick(seed, obs_idx):
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=seed)
    o = get_obstruction(obs, obs_idx)
    obs_x = o['x']
    obs_actual_top = o['y'] + o['height']   # y is bottom coord
    target_y = obs_actual_top + OBJ_TOP_OFFSET
    target_arm = PICK_ARM_JOINT
    print(f"Seed {seed} obs{obs_idx}: x={obs_x:.3f} y={o['y']:.3f} h={o['height']:.3f} actual_top={obs_actual_top:.3f}")
    print(f"  target_y={target_y:.3f} arm={target_arm:.3f} suction_y={target_y-target_arm-0.015:.4f}")

    for step in range(350):
        r = get_robot(obs)
        rx, ry, theta, arm = r['x'], r['y'], r['theta'], r['arm_joint']
        err = normalize_angle(-np.pi/2 - theta)
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
        dx = np.clip((obs_x - rx) * K_POS, -MAX_DX, MAX_DX)
        
        if step < 80:
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY); darm=0; vac=0
        elif step < 160:
            dy = np.clip((target_y - ry) * K_POS, -MAX_DY, MAX_DY); darm=0; vac=0
        elif step < 220:
            dy = np.clip((target_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((target_arm - arm) * 8, -0.1, 0.1); vac=0
        elif step < 280:
            dy = np.clip((target_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((target_arm - arm) * 8, -0.1, 0.1); vac=1.0
        else:
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((0.10 - arm) * 8, -0.1, 0.1); vac=1.0
        
        action = np.array([dx, dy, dtheta, darm, vac], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        
        if not obstruction_overlaps_surface(obs, obs_idx):
            r2 = get_robot(obs)
            o2 = get_obstruction(obs, obs_idx)
            print(f"  CLEARED at step {step}! robot=({r2['x']:.3f},{r2['y']:.3f}) obs_y={o2['y']:.3f}")
            return True
        if step in [159, 219, 279]:
            r2 = get_robot(obs)
            o2 = get_obstruction(obs, obs_idx)
            suc_y = r2['y'] - r2['arm_joint'] - 0.015
            print(f"  step={step} robot=({r2['x']:.3f},{r2['y']:.4f}) arm={r2['arm_joint']:.4f} suc_y={suc_y:.4f} obs_y={o2['y']:.4f}")
        if terminated or truncated: break
    return False

for seed in [0, 1, 2]:
    for obs_idx in [0, 1, 2, 3]:
        from obs_helpers import obstruction_overlaps_surface
        env2 = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
        obs2, _ = env2.reset(seed=seed)
        if obstruction_overlaps_surface(obs2, obs_idx):
            test_pick(seed, obs_idx)
            print()
