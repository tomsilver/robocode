import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import SHELF_FEATURES

env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, _ = env.reset(seed=0)

# Print all shelf features
base = 9
for i, fname in enumerate(SHELF_FEATURES):
    print(f"  shelf[{i}] {fname}: {obs[base+i]:.4f}")
