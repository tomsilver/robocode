"""Determine exact obs top by finding where arm stops during descent."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, obstruction_overlaps_surface
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA, K_POS, MAX_DX, MAX_DY, NAV_HIGH_Y
import numpy as np

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

# Move to x CLEAR of obs0 (x=0.20), descend to obs0_y level, then move right slowly
# Obs0: x=0.491, left_edge=0.434 (if center), width=0.114
obs_x = get_obstruction(obs, 0)['x']
print(f"obs0 x={obs_x}")

# First navigate to clear x, low y
for step in range(400):
    r = get_robot(obs)
    rx, ry, theta, arm = r['x'], r['y'], r['theta'], r['arm_joint']
    err = normalize_angle(-np.pi/2 - theta)
    dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
    
    if step < 80:  # nav to x=0.25, high y
        dx = np.clip((0.25 - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
    elif step < 180:  # descend to y=0.20
        dx = np.clip((0.25 - rx) * K_POS, -MAX_DX, MAX_DX)
        dy = np.clip((0.15 - ry) * K_POS, -MAX_DY, MAX_DY)  # go very low
    else:  # move right toward obs
        dx = 0.05  # max right
        dy = np.clip((0.20 - ry) * K_POS, -MAX_DY, MAX_DY)  # hold low y
    
    action = np.array([dx, dy, dtheta, 0, 0], dtype=np.float32)
    obs, _, terminated, truncated, _ = env.step(action)
    r2 = get_robot(obs)
    
    if step >= 180 and step % 5 == 0:
        print(f"  step={step} robot=({r2['x']:.3f},{r2['y']:.4f}) theta={r2['theta']:.3f}")
    if step >= 180 and r2['x'] > 0.60:
        break
    if terminated or truncated: break
