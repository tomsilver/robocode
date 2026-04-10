# Test: manually try to extend arm at the stuck position
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, extract_obstruction

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

# Manually teleport-ish: run some steps to get to EXTEND_DROP state
# Actually just test: start fresh and manually extend arm with vacuum ON
# Start state: let approach run to step 70 then take over
from approach import GeneratedApproach
approach = GeneratedApproach(env.action_space, env.observation_space, {})
approach.reset(obs, info)

for step in range(70):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)

r = extract_robot(obs)
o0 = extract_obstruction(obs, 0)
print(f"State at step 70: robot=({r['x']:.3f},{r['y']:.3f}), arm={r['arm_joint']:.4f}, vac={r['vacuum']:.1f}")
print(f"  obs0: x={o0['x']:.3f}, y={o0['y']:.3f}, w={o0['width']:.3f}, h={o0['height']:.3f}")
print(f"  arm range: {env.observation_space.low[4]:.3f} to {env.observation_space.high[4]:.3f}")
print(f"  action space: {env.action_space.low} to {env.action_space.high}")

# Try extending arm manually
for i in range(5):
    action = np.array([0, 0, 0, 0.1, 1.0], dtype=np.float32)  # max arm extension
    obs, reward, terminated, truncated, info = env.step(action)
    r = extract_robot(obs)
    o0 = extract_obstruction(obs, 0)
    print(f"  manual extend {i}: arm={r['arm_joint']:.4f}, robot_y={r['y']:.3f}, obs0_y={o0['y']:.3f}")

# Test with vacuum OFF
print("\nRelease vacuum then try extending:")
for i in range(5):
    action = np.array([0, 0, 0, 0.1, 0.0], dtype=np.float32)  # vacuum off, arm extend
    obs, reward, terminated, truncated, info = env.step(action)
    r = extract_robot(obs)
    o0 = extract_obstruction(obs, 0)
    print(f"  vac off {i}: arm={r['arm_joint']:.4f}, robot_y={r['y']:.3f}, obs0_y={o0['y']:.3f}")
