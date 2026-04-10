"""Scan arm extension at various robot y positions."""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA
import numpy as np

o0_x = 0.491
for target_y in [0.30, 0.35, 0.40, 0.45, 0.50]:
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, _ = env.reset(seed=0)
    # Navigate to obs_x, target_y
    for step in range(200):
        r = get_robot(obs)
        err = normalize_angle(-np.pi/2 - r['theta'])
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
        dx = np.clip((o0_x - r['x']) * 3.0, -0.05, 0.05)
        dy = np.clip((target_y - r['y']) * 3.0, -0.05, 0.05)
        obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, 0, 0], dtype=np.float32))
    r = get_robot(obs)
    start_y = r['y']
    # Now try to extend arm
    for step in range(60):
        r = get_robot(obs)
        err = normalize_angle(-np.pi/2 - r['theta'])
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
        dx = np.clip((o0_x - r['x']) * 3.0, -0.05, 0.05)
        dy = np.clip((target_y - r['y']) * 3.0, -0.05, 0.05)
        obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, 0.1, 0], dtype=np.float32))
    r = get_robot(obs)
    print(f"target_y={target_y:.2f}: actual_y={r['y']:.4f} arm={r['arm_joint']:.4f} "
          f"arm_tip_y={r['y']-r['arm_joint']:.4f}")
