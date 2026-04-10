"""Test minimum descent with theta=-pi/2."""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA
import numpy as np

for target_x in [0.15, 0.30, 0.491, 0.55]:
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, _ = env.reset(seed=0)

    for step in range(500):
        r = get_robot(obs)
        err = normalize_angle(-np.pi/2 - r['theta'])
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
        dx = np.clip((target_x - r['x']) * 3.0, -0.05, 0.05)
        dy = -0.05  # max downward
        obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, 0, 0], dtype=np.float32))

    r = get_robot(obs)
    print(f"x={target_x:.3f}: final x={r['x']:.4f} y={r['y']:.4f} theta={r['theta']:.3f}")
    print(f"  body_bottom={r['y']-0.10:.4f} arm_tip_y={r['y']-r['arm_joint']:.4f}")
