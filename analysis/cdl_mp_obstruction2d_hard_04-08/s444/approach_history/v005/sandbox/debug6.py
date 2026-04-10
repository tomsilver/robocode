import numpy as np
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, OBS0_START, OBS_STRIDE, NUM_OBSTRUCTIONS

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

print("Raw obs values:")
for i in range(NUM_OBSTRUCTIONS):
    base = OBS0_START + i * OBS_STRIDE
    print(f"  obs{i}: raw obs[{base}:{base+10}] = {obs[base:base+10]}")
    o = get_obstruction(obs, i)
    print(f"    → x={o['x']:.4f}, y={o['y']:.4f}, w={o['width']:.4f}, h={o['height']:.4f}")
    print(f"    → x_range=[{o['x']-o['width']/2:.4f}, {o['x']+o['width']/2:.4f}]")
    print(f"    → y_range=[{o['y']-o['height']/2:.4f}, {o['y']+o['height']/2:.4f}]")

robot = get_robot(obs)
print(f"\nRobot: x={robot['x']:.4f}, y={robot['y']:.4f}, base_radius={robot['base_radius']:.4f}, arm={robot['arm_joint']:.4f}")
print(f"  Raw obs[0:9] = {obs[0:9]}")
