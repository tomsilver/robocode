"""Find min_y at obs0_x with different approaches."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA, K_POS, MAX_DX, MAX_DY, NAV_HIGH_Y
import numpy as np

def find_miny_at(target_x, target_theta=0.0, n_steps=200):
    """Navigate to target_x, theta=target_theta (not arm-down), then descend."""
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=0)
    
    for step in range(n_steps):
        r = get_robot(obs)
        rx, ry, theta, arm = r['x'], r['y'], r['theta'], r['arm_joint']
        err = normalize_angle(target_theta - theta)
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
        dx = np.clip((target_x - rx) * K_POS, -MAX_DX, MAX_DX)
        
        if step < 80:
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
        else:
            dy = -MAX_DY  # max descent
        
        action = np.array([dx, dy, dtheta, 0, 0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
    
    r = get_robot(obs)
    return r['x'], r['y'], r['theta']

print("Min y at various x and theta combos:")
for tx, tt in [(0.491, -np.pi/2), (0.491, 0.0), (0.491, np.pi/2), 
               (0.400, -np.pi/2), (0.300, -np.pi/2), (0.491, np.pi)]:
    rx, ry, rth = find_miny_at(tx, tt)
    print(f"  target_x={tx:.3f} theta={tt:.2f}: actual=({rx:.3f},{ry:.4f}) theta={rth:.3f}")
