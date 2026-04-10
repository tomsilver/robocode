"""Debug nav_low descent with arm=0.20."""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA, K_POS, MAX_DX, MAX_DY
import numpy as np

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, _ = env.reset(seed=0)
o0 = get_obstruction(obs, 0)
obs_x = o0['x']; obs_top = o0['y'] + o0['height']/2
pick_y = obs_top + 0.20 + 0.012

# Phase 1: nav to (obs_x, 0.50) arm=0.10
for step in range(80):
    r = get_robot(obs)
    err = normalize_angle(-np.pi/2 - r['theta'])
    dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
    dx = np.clip((obs_x - r['x']) * K_POS, -MAX_DX, MAX_DX)
    dy = np.clip((0.50 - r['y']) * K_POS, -MAX_DY, MAX_DY)
    obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, 0, 0], dtype=np.float32))

r = get_robot(obs)
print(f"After nav_high: x={r['x']:.4f} y={r['y']:.4f} arm={r['arm_joint']:.4f}")

# Phase 2: extend arm to 0.20 at y=0.50
for step in range(40):
    r = get_robot(obs)
    err = normalize_angle(-np.pi/2 - r['theta'])
    dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
    dx = np.clip((obs_x - r['x']) * K_POS, -MAX_DX, MAX_DX)
    dy = np.clip((0.50 - r['y']) * K_POS, -MAX_DY, MAX_DY)
    obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, 0.1, 0], dtype=np.float32))

r = get_robot(obs)
print(f"After extend: x={r['x']:.4f} y={r['y']:.4f} arm={r['arm_joint']:.4f}")

# Phase 3: descend step by step
print(f"Descending to pick_y={pick_y:.4f}...")
for step in range(100):
    r = get_robot(obs)
    old_y = r['y']; old_arm = r['arm_joint']
    err = normalize_angle(-np.pi/2 - r['theta'])
    dtheta = np.clip(err * K_THETA, -MAX_DTHETA, MAX_DTHETA)
    dx = np.clip((obs_x - r['x']) * K_POS, -MAX_DX, MAX_DX)
    dy = np.clip((pick_y - r['y']) * K_POS, -MAX_DY, MAX_DY)
    darm = np.clip((0.20 - r['arm_joint']) * 8.0, -0.1, 0.1)
    obs, _, _, _, _ = env.step(np.array([dx, dy, dtheta, darm, 0], dtype=np.float32))
    r2 = get_robot(obs)
    if old_y != r2['y'] or step < 5 or step % 10 == 9:
        print(f"  step={step:3d}: y {old_y:.4f}→{r2['y']:.4f} arm {old_arm:.4f}→{r2['arm_joint']:.4f} dy_cmd={dy:.4f}")
    if abs(r2['y'] - pick_y) < 0.01:
        print(f"  Reached pick_y!")
        break
    if old_y == r2['y'] and step > 5:
        print(f"  STUCK at y={r2['y']:.4f} (target={pick_y:.4f})")
        break
