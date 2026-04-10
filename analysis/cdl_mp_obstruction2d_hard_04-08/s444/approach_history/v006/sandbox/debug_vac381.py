"""Test: navigate to (obs0_x, 0.40), let arm extend to 0.20, apply vacuum."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, obstruction_overlaps_surface
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA, K_POS, MAX_DX, MAX_DY, NAV_HIGH_Y
import numpy as np

for seed in [0, 1, 2]:
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=seed)
    o0 = get_obstruction(obs, 0)
    obs0_top = o0['y'] + o0['height']/2
    obs0_x = o0['x']
    target_y = obs0_top + 0.22  # higher than 0.381 threshold

    print(f"\nSeed {seed}: obs0_x={obs0_x:.3f} obs0_top={obs0_top:.3f} target_y={target_y:.3f}")

    for step in range(400):
        r = get_robot(obs)
        rx, ry, theta, arm = r['x'], r['y'], r['theta'], r['arm_joint']
        
        err = normalize_angle(-np.pi/2 - theta)
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
        dx = np.clip((obs0_x - rx) * K_POS, -MAX_DX, MAX_DX)
        
        if step < 100:
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = 0; vac = 0
        elif step < 200:
            dy = np.clip((target_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((0.20 - arm) * 8, -0.1, 0.1); vac = 0
        elif step < 300:
            dy = np.clip((target_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((0.20 - arm) * 8, -0.1, 0.1); vac = 1.0
        else:
            # Retract and lift
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((0.10 - arm) * 8, -0.1, 0.1); vac = 1.0
        
        action = np.array([dx, dy, dtheta, darm, vac], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        
        if step in [199, 299, 399]:
            r2 = get_robot(obs)
            o = get_obstruction(obs, 0)
            suc_y = r2['y'] - (r2['arm_joint'] + 0.015)
            print(f"  step={step} robot=({r2['x']:.3f},{r2['y']:.3f}) arm={r2['arm_joint']:.3f} suc_y={suc_y:.3f} obs_y={o['y']:.3f} on={obstruction_overlaps_surface(obs,0)}")
        
        if not obstruction_overlaps_surface(obs, 0):
            r2 = get_robot(obs)
            print(f"  CLEARED at step={step}! robot=({r2['x']:.3f},{r2['y']:.3f})")
            break
        if terminated or truncated:
            print(f"  terminated at step={step}")
            break
