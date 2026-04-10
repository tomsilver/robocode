"""Find floor constraints empirically."""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, get_surface
import numpy as np

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)
s = get_surface(obs); o0 = get_obstruction(obs, 0)
print(f"Surface: x={s['x']:.3f} y={s['y']:.3f} w={s['width']:.3f} h={s['height']:.3f}")
print(f"  surface spans y=[{s['y']-s['height']/2:.3f}, {s['y']+s['height']/2:.3f}]")
print(f"Obs0: x={o0['x']:.3f} y={o0['y']:.3f} w={o0['width']:.3f} h={o0['height']:.3f}")
print(f"  obs0 spans y=[{o0['y']-o0['height']/2:.3f}, {o0['y']+o0['height']/2:.3f}]")

# Find minimum y at x=0.2 (away from obs)
for step in range(500):
    r = get_robot(obs)
    dx = np.clip((0.15 - r['x']) * 3.0, -0.05, 0.05)
    dy = -0.05  # max downward
    obs, _, _, _, _ = env.step(np.array([dx, dy, 0, 0, 0], dtype=np.float32))

r = get_robot(obs)
print(f"\nMin y at x≈0.15: robot_y={r['y']:.4f} arm={r['arm_joint']:.4f}")
print(f"  body_bottom={r['y']-0.10:.4f}")
print(f"  arm_tip_y={r['y']-r['arm_joint']:.4f}")
