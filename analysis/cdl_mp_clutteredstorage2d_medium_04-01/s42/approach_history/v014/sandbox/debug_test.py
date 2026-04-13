import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from obs_helpers import *
from act_helpers import *

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv("kinder/ClutteredStorage2D-b3-v0")
obs, info = env.reset(seed=0)

robot = extract_robot(obs)
shelf = extract_shelf_inner(obs)
b0 = extract_block(obs, 0)
b1 = extract_block(obs, 1)

print("Robot:", robot)
print("Shelf inner:", shelf)
print("Block0:", b0, "in_shelf:", is_block_in_shelf(obs, 0))
print("Block1:", b1, "in_shelf:", is_block_in_shelf(obs, 1))
print("Block2:", extract_block(obs, 2), "in_shelf:", is_block_in_shelf(obs, 2))
print("Outside blocks:", get_outside_block_indices(obs))

# Test picking block1
# navigate below it
goal_y = b1.y - 0.5
goal_x = b1.x
print(f"\nPick goal: ({goal_x:.3f}, {goal_y:.3f})")

# Simulate navigation
from approach import GeneratedApproach
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

print("\nCurrent behavior:", type(approach._current).__name__)

# Run 100 steps and see what happens
for i in range(100):
    act = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(act)
    if i < 5 or i % 20 == 0:
        robot = extract_robot(obs)
        b1 = extract_block(obs, 1)
        print(f"  Step {i}: robot=({robot.x:.3f},{robot.y:.3f}) theta={robot.theta:.3f} arm={robot.arm_joint:.3f} vac={robot.vacuum:.1f} b1=({b1.x:.3f},{b1.y:.3f}) holding={is_holding_block(obs,1)}")

print("\nFinal behavior:", type(approach._current).__name__)
