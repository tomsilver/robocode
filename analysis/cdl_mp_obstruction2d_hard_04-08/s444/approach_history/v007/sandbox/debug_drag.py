"""Test: pick obs0 (arm=0.20, y=0.381) then drag LEFT out of surface x-range."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, obstruction_overlaps_surface, get_surface
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA, K_POS, MAX_DX, MAX_DY, NAV_HIGH_Y
import numpy as np

def test_drag(seed, obs_idx, drag_x=0.25):
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=seed)
    o = get_obstruction(obs, obs_idx)
    s = get_surface(obs)
    obs_x = o['x']
    obs_top = o['y'] + o['height']/2  # center formula (maybe wrong but used for reference)
    pick_y = 0.40  # target higher, arm=0.20 will settle robot at obs-constraint
    
    print(f"Seed {seed} obs{obs_idx}: x={obs_x:.3f} obs_h={o['height']:.3f} surf_x=[{s['x']-s['width']/2:.3f},{s['x']+s['width']/2:.3f}]")
    
    for step in range(400):
        r = get_robot(obs)
        rx, ry, theta, arm = r['x'], r['y'], r['theta'], r['arm_joint']
        err = normalize_angle(-np.pi/2 - theta)
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
        
        if step < 60:  # Nav to obs_x, NAV_HIGH_Y
            dx = np.clip((obs_x - rx) * K_POS, -MAX_DX, MAX_DX)
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = 0; vac = 0
        elif step < 120:  # Extend arm at high y
            dx = np.clip((obs_x - rx) * K_POS, -MAX_DX, MAX_DX)
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((0.20 - arm) * 8, -0.1, 0.1); vac = 0
        elif step < 180:  # Descend to pick_y
            dx = np.clip((obs_x - rx) * K_POS, -MAX_DX, MAX_DX)
            dy = np.clip((pick_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((0.20 - arm) * 8, -0.1, 0.1); vac = 0
        elif step < 220:  # Vacuum on, hold position
            dx = np.clip((obs_x - rx) * K_POS, -MAX_DX, MAX_DX)
            dy = np.clip((pick_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((0.20 - arm) * 8, -0.1, 0.1); vac = 1.0
        elif step < 300:  # Drag LEFT (arm still extended down)
            dx = np.clip((drag_x - rx) * K_POS, -MAX_DX, MAX_DX)
            dy = np.clip((pick_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((0.20 - arm) * 8, -0.1, 0.1); vac = 1.0
        else:  # release
            dx = 0; dy = 0; darm = 0; vac = 0
        
        action = np.array([dx, dy, dtheta, darm, vac], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        
        if not obstruction_overlaps_surface(obs, obs_idx):
            r2 = get_robot(obs)
            o2 = get_obstruction(obs, obs_idx)
            print(f"  CLEARED at step {step}! robot=({r2['x']:.3f},{r2['y']:.3f}) obs=({o2['x']:.3f},{o2['y']:.3f})")
            return True
        
        if step in [179, 219, 259, 299]:
            r2 = get_robot(obs)
            o2 = get_obstruction(obs, obs_idx)
            suc_y = r2['y'] - r2['arm_joint'] - 0.015
            print(f"  step={step} robot=({r2['x']:.3f},{r2['y']:.4f}) arm={r2['arm_joint']:.4f} suc_y={suc_y:.4f} obs=({o2['x']:.3f},{o2['y']:.4f})")
        
        if terminated or truncated: break
    return False

# Test obs that are on surface
for seed in [0]:
    for obs_idx in [0, 3]:  # both on surface in seed 0
        test_drag(seed, obs_idx)
        print()
