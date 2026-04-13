import sys, os
sys.path.insert(0, '/sandbox')
os.chdir('/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import *
from act_helpers import make_action
import math

from approach import GeneratedApproach
from primitives.motion_planning import BiRRT
PRIMITIVES = {'BiRRT': BiRRT}

# Run until block1 placed and retracted
env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, _ = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, PRIMITIVES)
approach.reset(obs, {})

for step in range(200):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    r = extract_robot(obs)
    cur = approach._current
    if type(cur).__name__ == 'PickupBlock' and r.vacuum < 0.5 and step > 50:
        break

# Now test arm extension from DIFFERENT robot x positions
# Always at y=2.1, theta=pi/2
for test_x in [0.205, 0.236, 0.270, 0.300]:
    env2 = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
    obs2, _ = env2.reset(seed=0)
    approach2 = GeneratedApproach(env2.action_space, env2.observation_space, PRIMITIVES)
    approach2.reset(obs2, {})
    for step in range(200):
        action2 = approach2.get_action(obs2)
        obs2, _, _, _, _ = env2.step(action2)
        r2 = extract_robot(obs2)
        if type(approach2._current).__name__ == 'PickupBlock' and r2.vacuum < 0.5 and step > 50:
            break
    
    # Navigate to test_x, 2.1
    for step in range(100):
        r2 = extract_robot(obs2)
        dx = test_x - r2.x
        dy = 2.100 - r2.y
        if abs(dx)<0.01 and abs(dy)<0.01:
            break
        obs2, _, _, _, _ = env2.step(make_action(np.clip(dx,-0.05,0.05), np.clip(dy,-0.05,0.05), 0, 0, 0))
    
    # Extend arm
    max_arm = 0.200
    for step in range(50):
        obs2, _, _, _, _ = env2.step(make_action(0, 0, 0, 0.02, 1.0))
        r2 = extract_robot(obs2)
        max_arm = max(max_arm, r2.arm_joint)
        if r2.arm_joint >= max_arm and step > 5:
            break  # arm stopped moving
    
    print(f"robot.x={test_x:.3f}: max_arm={max_arm:.3f}")

