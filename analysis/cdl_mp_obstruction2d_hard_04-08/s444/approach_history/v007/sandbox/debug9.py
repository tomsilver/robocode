import numpy as np
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_surface

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

print("Surface:", get_surface(obs))
print("obs[9:19]:", obs[9:19])

# What's the minimum y at x=0.20? The arm shaft hits something.
# What IS at the arm shaft bottom?
# arm_shaft_bottom = robot_y - arm_joint = 0.23076 - 0.10 = 0.13076
# Check if table info is available
print("\nFull obs (first 70):", obs[:70])
