"""Test standard pick (same as block picking) on obs0."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, obstruction_overlaps_surface, pick_robot_y
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA, K_POS, MAX_DX, MAX_DY, NAV_HIGH_Y, PICK_ARM_JOINT
import numpy as np

for seed in [0, 1, 2]:
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=seed)
    
    # Find first obstruction on surface
    from obs_helpers import NUM_OBSTRUCTIONS, obstruction_overlaps_surface
    obs_idx = next((i for i in range(NUM_OBSTRUCTIONS) if obstruction_overlaps_surface(obs, i)), None)
    if obs_idx is None:
        print(f"Seed {seed}: no obs on surface"); continue
    
    o = get_obstruction(obs, obs_idx)
    obs_x = o['x']
    target_y = pick_robot_y(o['y'], o['height'])  # = obs_top + 0.14
    target_arm = PICK_ARM_JOINT  # = 0.13
    obs_top = o['y'] + o['height']/2
    print(f"\nSeed {seed} obs{obs_idx}: x={obs_x:.3f} top={obs_top:.3f} target_y={target_y:.3f} arm={target_arm:.3f}")
    print(f"  suction will be at: {target_y - target_arm - 0.015:.4f} (want < {obs_top:.4f})")

    for step in range(350):
        r = get_robot(obs)
        rx, ry, theta, arm = r['x'], r['y'], r['theta'], r['arm_joint']
        err = normalize_angle(-np.pi/2 - theta)
        dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
        dx = np.clip((obs_x - rx) * K_POS, -MAX_DX, MAX_DX)
        
        if step < 80:  # nav to high
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = 0; vac = 0
        elif step < 160:  # nav to pick position
            dy = np.clip((target_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = 0; vac = 0
        elif step < 220:  # extend arm
            dy = np.clip((target_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((target_arm - arm) * 8, -0.1, 0.1); vac = 0
        elif step < 280:  # vacuum on
            dy = np.clip((target_y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((target_arm - arm) * 8, -0.1, 0.1); vac = 1.0
        else:  # lift
            dy = np.clip((NAV_HIGH_Y - ry) * K_POS, -MAX_DY, MAX_DY)
            darm = np.clip((0.10 - arm) * 8, -0.1, 0.1); vac = 1.0
        
        action = np.array([dx, dy, dtheta, darm, vac], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        
        if step in [159, 219, 279, 349]:
            r2 = get_robot(obs)
            o2 = get_obstruction(obs, obs_idx)
            suc_y = r2['y'] - r2['arm_joint'] - 0.015
            print(f"  step={step} robot=({r2['x']:.3f},{r2['y']:.4f}) arm={r2['arm_joint']:.4f} suc_y={suc_y:.4f} obs_y={o2['y']:.4f} on={obstruction_overlaps_surface(obs,obs_idx)}")
        
        if not obstruction_overlaps_surface(obs, obs_idx):
            print(f"  CLEARED at step {step}!")
            break
        if terminated or truncated:
            print(f"  terminated at step {step}")
            break
