"""Test pure descent at x=0.5224 vs pure x-movement."""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot
import numpy as np

def test_move(dx, dy, label):
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, _ = env.reset(seed=0)
    # Get to (0.5224, 0.4308) with arm=0.20
    from act_helpers import K_POS, MAX_DX, MAX_DY, normalize_angle, K_THETA, MAX_DTHETA
    for step in range(50):
        r = get_robot(obs)
        err = normalize_angle(-np.pi/2 - r['theta'])
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
        ddx = np.clip((0.4912 - r['x']) * K_POS, -MAX_DX, MAX_DX)
        ddy = np.clip((0.50 - r['y']) * K_POS, -MAX_DY, MAX_DY)
        obs, _, _, _, _ = env.step(np.array([ddx, ddy, dtheta, 0.1, 0], dtype=np.float32))
    # Now take ONE step with given action
    r = get_robot(obs)
    print(f"\n  Before {label}: x={r['x']:.4f} y={r['y']:.4f} arm={r['arm_joint']:.4f}")
    obs, _, _, _, _ = env.step(np.array([dx, dy, 0, 0, 0], dtype=np.float32))
    r = get_robot(obs)
    print(f"  After  {label}: x={r['x']:.4f} y={r['y']:.4f} arm={r['arm_joint']:.4f}")

test_move(0, -0.05, "dy=-0.05 dx=0")
test_move(-0.05, 0, "dy=0 dx=-0.05")
test_move(-0.05, -0.05, "dy=-0.05 dx=-0.05")
test_move(0, -0.05, "dy=-0.05 dx=0 (2nd)")
