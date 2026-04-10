"""Determine exact obs0 geometry by trying different y/arm combos for vacuum pickup."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, obstruction_overlaps_surface
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA, K_POS, MAX_DX, MAX_DY, NAV_HIGH_Y
import numpy as np

def test_vac(seed, obs_idx, target_y, target_arm):
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=seed)
    o = get_obstruction(obs, obs_idx)
    obs_top_computed = o['y'] + o['height']/2
    obs_x = o['x']

    for step in range(350):
        r = get_robot(obs)
        rx, ry, theta, arm = r['x'], r['y'], r['theta'], r['arm_joint']
        err = normalize_angle(-np.pi/2 - theta)
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
        dx = np.clip((obs_x - rx) * K_POS, -MAX_DX, MAX_DX)
        
        if step < 80:
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = 0; vac = 0
        elif step < 160:
            dy = np.clip((target_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((target_arm - arm) * 8, -0.1, 0.1); vac = 0
        elif step < 260:
            dy = np.clip((target_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((target_arm - arm) * 8, -0.1, 0.1); vac = 1.0
        else:
            # lift
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = 0; vac = 1.0
        
        action = np.array([dx, dy, dtheta, darm, vac], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        
        if not obstruction_overlaps_surface(obs, obs_idx):
            print(f"  CLEARED! seed={seed} obs{obs_idx} target_y={target_y:.3f} arm={target_arm:.3f} at step={step}")
            return True
        if terminated or truncated: break
    
    r = get_robot(obs)
    o = get_obstruction(obs, obs_idx)
    suc_y = r['y'] - r['arm_joint'] - 0.015
    print(f"  FAIL seed={seed} obs{obs_idx} target_y={target_y:.3f} arm={target_arm:.3f}: "
          f"robot_y={r['y']:.4f} arm={r['arm_joint']:.4f} suc_y={suc_y:.4f} obs_top_comp={obs_top_computed:.4f} obs_y={o['y']:.4f}")
    return False

# Test with different combinations for obs0 (seed 0, obs_top_computed=0.161)
print("=== obs0 (seed 0, computed_top=0.161) ===")
for ty, ta in [(0.40, 0.20), (0.42, 0.20), (0.44, 0.20), (0.40, 0.18), (0.45, 0.20), (0.50, 0.20)]:
    test_vac(0, 0, ty, ta)

print("\n=== obs3 (seed 0) ===")
env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)
o3 = get_obstruction(obs, 3)
print(f"obs3: x={o3['x']:.3f} y={o3['y']:.4f} w={o3['width']:.4f} h={o3['height']:.4f} computed_top={o3['y']+o3['height']/2:.4f}")
for ty, ta in [(0.40, 0.20), (0.44, 0.20), (0.48, 0.20)]:
    test_vac(0, 3, ty, ta)
