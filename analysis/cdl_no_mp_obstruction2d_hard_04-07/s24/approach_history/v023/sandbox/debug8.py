import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, extract_target_block

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, {})
approach.reset(obs, info)

for step in range(50):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)

# At step 50, robot should be stuck at y=0.500
r = extract_robot(obs)
print(f"Robot: ({r['x']:.3f}, {r['y']:.3f}), arm={r['arm_joint']:.3f}, vac={r['vacuum']:.1f}")
for i in range(4):
    o = extract_obstruction(obs, i)
    print(f"obs{i}: x={o['x']:.3f}, y={o['y']:.3f}, w={o['width']:.3f}, h={o['height']:.3f}, top={o['y']+o['height']:.3f}")
blk = extract_target_block(obs)
print(f"block: x={blk['x']:.3f}, y={blk['y']:.3f}, h={blk['height']:.3f}")
print(f"\nRobot base lower bound: {r['y'] - r['base_radius']:.3f}")
print(f"Gripper bottom: {r['y'] - r['arm_joint']:.3f}")
print(f"Obs0 top: {extract_obstruction(obs,0)['y'] + extract_obstruction(obs,0)['height']:.3f}")

# Try descending manually
print("\nTrying manual descent:")
for i in range(10):
    action = np.array([0, -0.05, 0, 0, 1.0], dtype=np.float32)
    obs_before = obs.copy()
    obs, reward, terminated, truncated, info = env.step(action)
    r = extract_robot(obs)
    o0 = extract_obstruction(obs, 0)
    print(f"  step {i}: robot_y={r['y']:.4f}, obs0_y={o0['y']:.4f}")

# Reset and test: what if we just navigate to (1.438, 0.412) directly without carrying anything?
print("\n--- Test: navigate to drop zone low with vacuum OFF ---")
env2 = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs2, _ = env2.reset(seed=0)
# Just fly robot to x=1.438, y=0.412 without picking anything
for step in range(200):
    r2 = extract_robot(obs2)
    dx = np.clip(1.438 - r2['x'], -0.05, 0.05)
    dy = np.clip(0.412 - r2['y'], -0.05, 0.05)
    action = np.array([dx, dy, 0, 0, 0], dtype=np.float32)
    obs2, _, _, _, _ = env2.step(action)
    r2 = extract_robot(obs2)
    if abs(r2['x'] - 1.438) < 0.01 and abs(r2['y'] - 0.412) < 0.01:
        print(f"Reached (1.438, 0.412) at step {step}")
        break
    if step % 20 == 0:
        print(f"  step {step}: ({r2['x']:.3f}, {r2['y']:.3f})")
else:
    print(f"Failed, ended at ({r2['x']:.3f}, {r2['y']:.3f})")
