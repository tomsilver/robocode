# Test arm extension from initial state
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)
r = extract_robot(obs)
print(f"Initial: arm={r['arm_joint']:.4f}, arm_length={r['arm_length']:.4f}, base_radius={r['base_radius']:.4f}")

# Try extending arm from initial state
for i in range(10):
    action = np.array([0, 0, 0, 0.1, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    r = extract_robot(obs)
    print(f"  step {i}: arm={r['arm_joint']:.4f}")
