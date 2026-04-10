import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import *
from act_helpers import *

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}
obs, info = env.reset(seed=42)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

# Simulate until NAV_DOWN
for step in range(25):
    action = approach.get_action(obs)
    obs, _, _, _, _ = env.step(action)

robot = extract_robot(obs)
block = extract_target_block(obs)
print(f"At step 25: robot=({robot['x']:.4f},{robot['y']:.4f}) arm={robot['arm_joint']:.3f}")
print(f"Block: ({block['x']:.4f},{block['y']:.4f}) w={block['width']:.3f} h={block['height']:.3f}")
print(f"Phase: {approach._current._phase}")
print(f"pick_x={approach._current._pick_x:.4f}, pick_y={approach._current._pick_y:.4f}")

# Try to navigate to pick_y manually
print("\nManual descent test:")
for i in range(30):
    action = np.zeros(5); action[1] = -0.05  # go down
    prev_y = extract_robot(obs)['y']
    obs, _, _, _, _ = env.step(action)
    new_y = extract_robot(obs)['y']
    if abs(prev_y - new_y) < 0.001:
        print(f"  Step {i}: STUCK at y={new_y:.4f} (tried -{0.05:.2f})")
    else:
        print(f"  Step {i}: y={prev_y:.4f} -> {new_y:.4f}")
    if new_y < approach._current._pick_y + 0.02:
        print("  Reached pick_y!")
        break
