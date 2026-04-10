"""Debug why robot can't descend at x=0.5224."""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, NUM_OBSTRUCTIONS
from behaviors import PickOneObstruction
import numpy as np

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, _ = env.reset(seed=0)
b = PickOneObstruction(None, 0)
b.reset(obs)

for step in range(12):
    obs, _, _, _, _ = env.step(b.step(obs))

r = get_robot(obs)
print(f"At step 12: x={r['x']:.4f} y={r['y']:.4f} arm={r['arm_joint']:.4f}")
for i in range(NUM_OBSTRUCTIONS):
    o = get_obstruction(obs, i)
    top = o['y'] + o['height']/2
    print(f"  Obs{i}: x={o['x']:.4f} top={top:.4f} spans_x=[{o['x']-o['width']/2:.4f},{o['x']+o['width']/2:.4f}]")

# Try moving left to x=0.491 first
print("\nMoving left to obs_x before descent:")
for step in range(20):
    r = get_robot(obs)
    dx = np.clip((0.491 - r['x']) * 3.0, -0.05, 0.05)
    obs, _, _, _, _ = env.step(np.array([dx, 0, 0, 0, 0], dtype=np.float32))
r = get_robot(obs)
print(f"After x-align: x={r['x']:.4f} y={r['y']:.4f}")

# Now try descending
print("Descending:")
for step in range(10):
    r = get_robot(obs)
    obs, _, _, _, _ = env.step(np.array([0, -0.05, 0, 0, 0], dtype=np.float32))
    r2 = get_robot(obs)
    print(f"  step={step}: y {r['y']:.4f}→{r2['y']:.4f}")
