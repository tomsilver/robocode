"""Find exact min_y at obs0_x when arm=0.20 extended."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, obstruction_overlaps_surface
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA, K_POS, MAX_DX, MAX_DY, NAV_HIGH_Y
import numpy as np

def test(seed, obs_idx, target_y):
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=seed)
    o = get_obstruction(obs, obs_idx)
    obs_top = o['y'] + o['height']/2
    obs_x = o['x']

    for step in range(250):
        r = get_robot(obs)
        rx, ry, theta, arm = r['x'], r['y'], r['theta'], r['arm_joint']
        err = normalize_angle(-np.pi/2 - theta)
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
        dx = np.clip((obs_x - rx) * K_POS, -MAX_DX, MAX_DX)
        if step < 80:
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = 0; vac = 0
        elif step < 160:
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((0.20 - arm) * 8, -0.1, 0.1); vac = 0
        else:
            # Descend with arm extended
            dy = np.clip((target_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((0.20 - arm) * 8, -0.1, 0.1); vac = 0
        action = np.array([dx, dy, dtheta, darm, vac], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
    
    r = get_robot(obs)
    suc_y = r['y'] - r['arm_joint'] - 0.015
    print(f"  seed={seed} obs{obs_idx} target_y={target_y:.3f}: robot_y={r['y']:.4f} arm={r['arm_joint']:.4f} suc_y={suc_y:.4f} obs_top={obs_top:.4f} gap={suc_y-obs_top:.4f}")

# obs0, seed 0: obs_top=0.161, want suc_y=0.161 -> robot_y=0.376
for target_y in [0.376, 0.38, 0.39, 0.40, 0.42]:
    test(0, 0, target_y)
