import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, IDX_ROBOT_BR, IDX_ROBOT_AJ, IDX_ROBOT_AL

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=1)
r = extract_robot(obs)
print("Robot info:")
for k, v in r.items():
    print(f"  {k} = {v}")
print(f"\nObs indices: BR={IDX_ROBOT_BR}, AJ={IDX_ROBOT_AJ}, AL={IDX_ROBOT_AL}")
print(f"Raw obs values at robot indices: {obs[0:9]}")
