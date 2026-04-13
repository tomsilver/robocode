import sys, os
sys.path.insert(0, '/sandbox')
os.chdir('/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import *
from act_helpers import make_action
import math

# Simulate: after placing block1, try to place block2 manually
# First, run until block1 is placed and retracted
from approach import GeneratedApproach
from primitives.motion_planning import BiRRT
PRIMITIVES = {'BiRRT': BiRRT}

env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, _ = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, PRIMITIVES)
approach.reset(obs, {})

for step in range(150):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, _ = env.step(action)

r = extract_robot(obs)
print(f"After 150 steps: robot=({r.x:.3f},{r.y:.3f}) arm={r.arm_joint:.3f}")
for name in BLOCK_NAMES:
    cx, cy = get_block_center(obs, name)
    print(f"  {name}: ({cx:.3f},{cy:.3f}) in={is_block_in_shelf(obs, name)}")

# Now manually position robot at (0.222, 2.1), theta=pi/2, arm=0.2 with a block picked up
# Just extend arm from 0.200 to max with vacuum ON from current state
print("\nRaw arm extension test (vacuum=1, arm starting wherever it is):")
r = extract_robot(obs)
print(f"Starting: arm={r.arm_joint:.3f} robot=({r.x:.3f},{r.y:.3f}) theta={r.theta:.3f}")
for step in range(50):
    obs, _, _, _, _ = env.step(make_action(0, 0, 0, 0.02, 1.0))
    r = extract_robot(obs)
    if step % 5 == 4 or r.arm_joint < 0.205 or r.arm_joint > 0.395:
        b1cx, b1cy = get_block_center(obs, 'block1')
        b0cx, b0cy = get_block_center(obs, 'block0')
        print(f"  s={step} arm={r.arm_joint:.3f} r=({r.x:.3f},{r.y:.3f}) b1=({b1cx:.3f},{b1cy:.3f}) b0=({b0cx:.3f},{b0cy:.3f})")

