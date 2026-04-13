import sys, os
sys.path.insert(0, '/sandbox')
os.chdir('/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import *
from primitives.motion_planning import BiRRT
from act_helpers import make_action
PRIMITIVES = {'BiRRT': BiRRT}

env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, info = env.reset(seed=0)

# Manually navigate: robot to (0.236, 2.1), theta=pi/2, then extend arm
import math

# Set robot directly via reset to a convenient position
# Use make_action to navigate manually
r = extract_robot(obs)
print(f"Robot: ({r.x:.3f},{r.y:.3f})")

# Read block0 position in the shelf
b0_cx, b0_cy = get_block_center(obs, 'block0')
print(f"block0 center: ({b0_cx:.3f},{b0_cy:.3f})")

# For now just run approach and check block1 after placement
from approach import GeneratedApproach
env2 = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs2, _ = env2.reset(seed=0)
approach = GeneratedApproach(env2.action_space, env2.observation_space, PRIMITIVES)
approach.reset(obs2, {})

# Run until block1 placed
b1_in = False
for step in range(200):
    action = approach.get_action(obs2)
    obs2, reward, terminated, truncated, _ = env2.step(action)
    if is_block_in_shelf(obs2, 'block1') and not b1_in:
        b1_in = True
        cx, cy = get_block_center(obs2, 'block1')
        b0_cx, b0_cy = get_block_center(obs2, 'block0')
        print(f"  s={step}: block1 placed! center=({cx:.3f},{cy:.3f}), block0=({b0_cx:.3f},{b0_cy:.3f})")
        # Wait 20 more steps and check if positions settle
        break

for step2 in range(20):
    obs2, _, _, _, _ = env2.step(make_action(0,0,0,0,0))

cx, cy = get_block_center(obs2, 'block1')
b0cx, b0cy = get_block_center(obs2, 'block0')
print(f"  After settling: block1=({cx:.3f},{cy:.3f}), block0=({b0cx:.3f},{b0cy:.3f})")

# Now try to place block2: extend arm slowly from 0.200 to max
r = extract_robot(obs2)
# First navigate back to shelf position manually
for step in range(50):
    obs2, _, _, _, _ = env2.step(make_action(0,0,0,0,0))

print("\nExtending arm slowly (vacuum ON):")
r = extract_robot(obs2)
print(f"Robot at ({r.x:.3f},{r.y:.3f}) theta={r.theta:.3f} arm={r.arm_joint:.3f}")
for step in range(40):
    obs2, _, _, _, _ = env2.step(make_action(0,0,0,0.02,1.0))  # extend arm
    r = extract_robot(obs2)
    cx,cy = get_block_center(obs2, 'block1')
    b0cx,b0cy = get_block_center(obs2, 'block0')
    if step % 5 == 0:
        print(f"  s={step} arm={r.arm_joint:.3f} b1=({cx:.3f},{cy:.3f}) b0=({b0cx:.3f},{b0cy:.3f})")

