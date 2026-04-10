"""Test robot descent at x=0.2 (no obs nearby)."""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot
from act_helpers import K_POS, MAX_DX, MAX_DY
import numpy as np

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

for step in range(200):
    r = get_robot(obs)
    rx, ry = r['x'], r['y']
    dx = np.clip((0.2 - rx) * K_POS, -MAX_DX, MAX_DX)
    dy = np.clip((0.3 - ry) * K_POS, -MAX_DY, MAX_DY)
    action = np.array([dx, dy, 0, 0, 0], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)

r = get_robot(obs)
print(f"At x=0.2, target_y=0.3: actual_y={r['y']:.4f} actual_x={r['x']:.4f}")

env2 = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs2, _ = env2.reset(seed=0)
for step in range(200):
    r = get_robot(obs2)
    rx, ry = r['x'], r['y']
    dx = np.clip((0.491 - rx) * K_POS, -MAX_DX, MAX_DX)
    dy = np.clip((0.2 - ry) * K_POS, -MAX_DY, MAX_DY)
    action = np.array([dx, dy, 0, 0, 0], dtype=np.float32)
    obs2, _, _, _, _ = env2.step(action)
r = get_robot(obs2)
print(f"At x=0.491 (obs_x), target_y=0.2: actual_y={r['y']:.4f} actual_x={r['x']:.4f}")
