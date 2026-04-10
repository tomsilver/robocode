"""Test from exact position (0.5224, 0.4308) with arm=0.20."""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction
from act_helpers import K_POS, MAX_DX, MAX_DY, normalize_angle, K_THETA, MAX_DTHETA
import numpy as np

# Reproduce exact state from debug_navlow (step 10 result)
env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, _ = env.reset(seed=0)

b_target = (0.5224, 0.4308)
# first get to near start position of behavior
from behaviors import PickOneObstruction
b = PickOneObstruction(None, 0)
b.reset(obs)
for step in range(11):
    action = b.step(obs)
    obs, _, _, _, _ = env.step(action)

r = get_robot(obs)
print(f"Reproduced: x={r['x']:.4f} y={r['y']:.4f} arm={r['arm_joint']:.4f} theta={r['theta']:.4f}")
print(f"Phase: {b._phase}")

# Now test actions from this state
obs2 = obs  # save
for dx_t, dy_t in [(0, -0.05), (-0.05, 0), (-0.05, -0.05)]:
    env2 = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    # can't easily copy state, so just use the same obs
    before_r = get_robot(obs)
    obs_t = obs  
    # just apply action
    obs_t, _, _, _, _ = env.step(np.array([dx_t, dy_t, 0, 0, 0], dtype=np.float32))
    after_r = get_robot(obs_t)
    print(f"  dx={dx_t:.2f} dy={dy_t:.2f}: x {before_r['x']:.4f}→{after_r['x']:.4f} "
          f"y {before_r['y']:.4f}→{after_r['y']:.4f}")
    # restore - we can't, so let's track what happened
    obs = obs_t  # unfortunately state changes
    before_r = after_r
