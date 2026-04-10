"""Test descent at obs_x with arm=0.20."""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, obstruction_overlaps_surface
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA, K_POS, MAX_DX, MAX_DY
import numpy as np

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, _ = env.reset(seed=0)
o0 = get_obstruction(obs, 0)
obs_x = o0['x']; obs_top = o0['y'] + o0['height']/2
pick_y = obs_top + 0.20 + 0.012

# Phase 1: nav to (obs_x, 0.50) - TIGHT x
for step in range(150):
    r = get_robot(obs)
    err = normalize_angle(-np.pi/2 - r['theta'])
    dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
    dx = np.clip((obs_x - r['x']) * K_POS, -MAX_DX, MAX_DX)
    dy = np.clip((0.50 - r['y']) * K_POS, -MAX_DY, MAX_DY)
    obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, 0, 0], dtype=np.float32))

r = get_robot(obs)
print(f"After nav_high (150 steps): x={r['x']:.4f} y={r['y']:.4f}")

# Phase 2: extend arm at same height (NO downward movement)
for step in range(30):
    r = get_robot(obs)
    err = normalize_angle(-np.pi/2 - r['theta'])
    dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
    dx = np.clip((obs_x - r['x']) * K_POS, -MAX_DX, MAX_DX)
    dy = np.clip((0.50 - r['y']) * K_POS, -MAX_DY, MAX_DY)
    obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, 0.1, 0], dtype=np.float32))
r = get_robot(obs)
print(f"After extend: x={r['x']:.4f} y={r['y']:.4f} arm={r['arm_joint']:.4f}")

# Phase 3: PURE DESCENT - no x movement
print(f"Pure descent to pick_y={pick_y:.4f}:")
for step in range(30):
    r = get_robot(obs)
    old_y = r['y']
    dy = np.clip((pick_y - r['y']) * K_POS, -MAX_DY, MAX_DY)
    obs, _, _, _, _ = env.step(np.array([0, dy, 0, 0, 0], dtype=np.float32))
    r2 = get_robot(obs)
    if step < 5 or abs(r2['y'] - pick_y) < 0.005:
        print(f"  step={step}: y={r2['y']:.4f} dy_cmd={dy:.3f}")

r = get_robot(obs)
print(f"Final: x={r['x']:.4f} y={r['y']:.4f} arm={r['arm_joint']:.4f}")
print(f"arm_tip_y={r['y']-r['arm_joint']:.4f} (obs_top={obs_top:.4f})")

# Vacuum on
print("Applying vacuum...")
for step in range(15):
    obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0, 1.0], dtype=np.float32))

on = obstruction_overlaps_surface(obs, 0)
o0_new = get_obstruction(obs, 0)
print(f"After vacuum: obs0_y={o0_new['y']:.4f} still_on={on}")
