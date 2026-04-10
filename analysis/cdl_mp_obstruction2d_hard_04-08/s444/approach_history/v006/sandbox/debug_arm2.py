"""Test arm extension at various y levels above obs0."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_obstruction, get_robot
from act_helpers import clip_action, normalize_angle, K_THETA, K_POS, MAX_DX, MAX_DY, MAX_DTHETA
import numpy as np

for target_y in [0.38, 0.40, 0.45]:
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=0)

    obs0_x = 0.491
    NAV_HIGH_Y = 0.50

    # Navigate to (obs0_x, target_y) then extend arm
    for step in range(300):
        r = get_robot(obs)
        rx, ry, theta, arm = r['x'], r['y'], r['theta'], r['arm_joint']

        err = normalize_angle(-np.pi/2 - theta)
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)

        if step < 80:
            # Nav to high
            dx = np.clip((obs0_x - rx) * K_POS, -MAX_DX, MAX_DX)
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
            action = np.array([dx, dy, dtheta, 0, 0], dtype=np.float32)
        elif step < 160:
            # Descend to target_y
            dx = np.clip((obs0_x - rx) * K_POS, -MAX_DX, MAX_DX)
            dy = np.clip((target_y - ry) * K_POS, -MAX_DY, MAX_DY)
            action = np.array([dx, dy, dtheta, 0, 0], dtype=np.float32)
        else:
            # Extend arm
            dx = np.clip((obs0_x - rx) * K_POS, -MAX_DX, MAX_DX)
            dy = np.clip((target_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((0.20 - arm) * 8.0, -0.1, 0.1)
            action = np.array([dx, dy, dtheta, darm, 0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

    r = get_robot(obs)
    o0 = get_obstruction(obs, 0)
    print(f"target_y={target_y}: final robot=({r['x']:.3f},{r['y']:.3f}) arm={r['arm_joint']:.4f} theta={r['theta']:.3f}")
    print(f"  obs0=({o0['x']:.3f},{o0['y']:.3f})")
