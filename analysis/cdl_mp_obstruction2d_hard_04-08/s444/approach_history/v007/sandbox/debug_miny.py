"""Find minimum reachable y at various x positions."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot
from act_helpers import clip_action
import numpy as np

def find_min_y(seed, target_x, arm_up=False):
    """Navigate to target_x, then try to descend as low as possible."""
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=seed)

    from act_helpers import normalize_angle, K_THETA, K_POS, MAX_DX, MAX_DY, MAX_DTHETA, NAV_HIGH_Y

    for step in range(300):
        r = get_robot(obs)
        rx, ry, theta, arm = r['x'], r['y'], r['theta'], r['arm_joint']

        if arm_up:
            target_theta = np.pi / 2
        else:
            target_theta = -np.pi / 2

        err = normalize_angle(target_theta - theta)
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)

        if step < 100:
            # Go to (target_x, NAV_HIGH_Y)
            dx = np.clip((target_x - rx) * K_POS, -MAX_DX, MAX_DX)
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
        else:
            # Try to descend to y=0
            dx = np.clip((target_x - rx) * K_POS, -MAX_DX, MAX_DX)
            dy = -MAX_DY  # max downward speed

        action = np.array([dx, dy, dtheta, 0, 0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)

    r = get_robot(obs)
    return r['x'], r['y'], r['theta']

print("Minimum reachable y at various x positions:")
for x in [0.20, 0.30, 0.40, 0.49, 0.50, 0.60, 0.70]:
    rx, ry, theta = find_min_y(0, x, arm_up=False)
    rx_up, ry_up, theta_up = find_min_y(0, x, arm_up=True)
    print(f"  x={x:.2f}: arm_down y={ry:.4f} (actual_x={rx:.3f} theta={theta:.3f}) | arm_up y={ry_up:.4f}")
