"""Find min y at obs_x with robot already aligned."""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction
import numpy as np

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs_s, _ = env.reset(seed=0)
o0 = get_obstruction(obs_s, 0)
obs_x = o0['x']; obs_top = o0['y'] + o0['height']/2
print(f"obs0: x={obs_x:.3f} top={obs_top:.3f}")

# Phase 1: get to (obs_x, 0.50)
for step in range(100):
    r = get_robot(obs_s)
    dx = np.clip((obs_x - r['x']) * 3.0, -0.05, 0.05)
    dy = np.clip((0.50 - r['y']) * 3.0, -0.05, 0.05)
    obs_s, _, _, _, _ = env.step(np.array([dx, dy, 0, 0, 0], dtype=np.float32))
r = get_robot(obs_s)
print(f"After nav_high: x={r['x']:.4f} y={r['y']:.4f}")

# Phase 2: from high, try hard descent
for step in range(300):
    r = get_robot(obs_s)
    dx = np.clip((obs_x - r['x']) * 3.0, -0.05, 0.05)
    dy = -0.05  # max downward
    obs_s, _, _, _, _ = env.step(np.array([dx, dy, 0, 0, 0], dtype=np.float32))
    if step % 50 == 49:
        r = get_robot(obs_s)
        print(f"  step={step}: x={r['x']:.4f} y={r['y']:.4f}")

r = get_robot(obs_s)
print(f"Min y at obs_x: x={r['x']:.4f} y={r['y']:.4f}")
print(f"  body_bottom={r['y']-0.10:.4f} (obs_top={obs_top:.4f})")
