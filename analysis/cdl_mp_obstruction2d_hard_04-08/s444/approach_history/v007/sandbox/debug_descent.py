"""Pure y descent test: x fixed, theta=-pi/2, measure stuck y."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, obstruction_overlaps_surface
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA, NAV_HIGH_Y
import numpy as np

def test_descent(seed, obs_idx, target_arm=0.185, slow=True):
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=seed)

    from obs_helpers import get_obstruction
    o = get_obstruction(obs, obs_idx)
    obs_top = o['y'] + o['height']/2
    obs_x = o['x']
    pick_y = obs_top + 0.20
    print(f"  obs{obs_idx}: x={obs_x:.3f} top={obs_top:.4f} pick_y={pick_y:.4f} arm={target_arm}")

    for step in range(500):
        r = get_robot(obs)
        rx, ry, theta, arm = r['x'], r['y'], r['theta'], r['arm_joint']

        err = normalize_angle(-np.pi/2 - theta)
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)

        if step < 80:
            # Navigate to (obs_x, NAV_HIGH_Y)
            dx = np.clip((obs_x - rx) * 3.0, -0.05, 0.05)
            dy = np.clip((NAV_HIGH_Y - ry) * 3.0, -0.05, 0.05)
            darm = 0
            vac = 0
        elif step < 160:
            # PURE Y descent with slow step, hold x fixed
            dx = 0  # No x movement
            if slow:
                dy = np.clip((pick_y - ry) * 1.5, -0.025, 0.025)
            else:
                dy = np.clip((pick_y - ry) * 3.0, -0.05, 0.05)
            darm = 0
            vac = 0
        elif step < 240:
            # Extend arm
            dx = 0
            dy = np.clip((pick_y - ry) * 1.5, -0.025, 0.025)
            darm = np.clip((target_arm - arm) * 8.0, -0.1, 0.1)
            vac = 0
        else:
            # Apply vacuum
            dx = 0
            dy = np.clip((pick_y - ry) * 1.5, -0.025, 0.025)
            darm = np.clip((target_arm - arm) * 8.0, -0.1, 0.1)
            vac = 1.0

        action = np.array([dx, dy, dtheta, darm, vac], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        r2 = get_robot(obs)

        if step in [79, 80, 120, 159, 160, 200, 239, 240, 260, 280, 300]:
            o0 = get_obstruction(obs, obs_idx)
            suc_y = r2['y'] + (r2['arm_joint']+0.015)*np.sin(r2['theta'])
            print(f"    step={step:3d} robot=({r2['x']:.3f},{r2['y']:.4f}) arm={r2['arm_joint']:.4f} "
                  f"theta={r2['theta']:.3f} suc_y={suc_y:.4f} obs=({o0['x']:.3f},{o0['y']:.4f}) "
                  f"on={obstruction_overlaps_surface(obs, obs_idx)}")

        if not obstruction_overlaps_surface(obs, obs_idx):
            print(f"    CLEARED at step {step}!")
            return True
        if terminated or truncated:
            break
    return False

print("=== Seed 0, obs0 (top=0.161) ===")
test_descent(0, 0)
print("\n=== Seed 0, obs3 (top=0.167) ===")
test_descent(0, 3)
