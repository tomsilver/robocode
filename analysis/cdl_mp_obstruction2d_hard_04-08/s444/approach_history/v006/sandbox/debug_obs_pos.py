"""Check obs positions when robot is stuck."""
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
    action = b.step(obs)
    obs, _, _, _, _ = env.step(action)

r = get_robot(obs)
print(f"Robot: x={r['x']:.4f} y={r['y']:.4f} arm={r['arm_joint']:.4f}")
for i in range(NUM_OBSTRUCTIONS):
    o = get_obstruction(obs, i)
    print(f"Obs{i}: x={o['x']:.4f} y={o['y']:.4f} w={o['width']:.4f} h={o['height']:.4f}")
    print(f"  spans x=[{o['x']-o['width']/2:.4f},{o['x']+o['width']/2:.4f}] y=[{o['y']-o['height']/2:.4f},{o['y']+o['height']/2:.4f}]")

# Now try just dy=-0.01
print(f"\nTrying small dy=-0.01 from stuck position:")
for dy in [-0.01, -0.005, -0.001]:
    obs_t = obs  # same state
    obs_t, _, _, _, _ = env.step(np.array([0, dy, 0, 0, 0], dtype=np.float32))
    r2 = get_robot(obs_t)
    print(f"  dy={dy}: y {r['y']:.4f}→{r2['y']:.4f} (moved={r2['y']!=r['y']})")
    # restore by re-running behavior steps (not ideal but testing)
