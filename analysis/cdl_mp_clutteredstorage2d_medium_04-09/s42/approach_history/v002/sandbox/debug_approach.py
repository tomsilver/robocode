"""Debug the approach step by step."""
import sys
sys.path.insert(0, '/sandbox')

import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from obs_helpers import extract_robot, extract_rect, is_block_in_shelf, get_gripper_pos, blocks_outside_shelf

primitives = {'BiRRT': BiRRT}

env = KinderGeom2DEnv("kinder/ClutteredStorage2D-b3-v0")
obs, info = env.reset(seed=0)

print("Initial state:")
robot = extract_robot(obs)
print(f"  Robot: x={robot.x:.3f} y={robot.y:.3f} theta={robot.theta:.3f} arm_joint={robot.arm_joint:.3f} vacuum={robot.vacuum:.1f}")
for name in ['block0','block1','block2']:
    b = extract_rect(obs, name)
    inside = is_block_in_shelf(obs, name)
    print(f"  {name}: x={b.x:.3f} y={b.y:.3f} theta={b.theta:.3f} in_shelf={inside}")

print("\nOutside blocks:", blocks_outside_shelf(obs))

# Test single pick behavior
from behaviors import PickBlock, PlaceInShelf

block_name = 'block1'
pick = PickBlock(block_name, primitives)
pick.reset(obs)
print(f"\nPickBlock({block_name}) initializable: {pick.initializable(obs)}")
print(f"PickBlock({block_name}) terminated: {pick.terminated(obs)}")
print(f"Phase: {pick._phase}, actions: {len(pick._actions)}")
if pick._actions:
    print(f"First action: {list(pick._actions)[0]}")

# Simulate pick
for step in range(200):
    if pick.terminated(obs):
        print(f"  Picked block at step {step}!")
        robot = extract_robot(obs)
        block = extract_rect(obs, block_name)
        gx, gy = get_gripper_pos(robot)
        print(f"  Gripper: ({gx:.3f}, {gy:.3f}), Block: ({block.x:.3f}, {block.y:.3f})")
        break
    action = pick.step(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if step % 50 == 0:
        robot = extract_robot(obs)
        block = extract_rect(obs, block_name)
        gx, gy = get_gripper_pos(robot)
        print(f"  Step {step}: robot=({robot.x:.3f},{robot.y:.3f}) arm={robot.arm_joint:.3f} theta={robot.theta:.3f} vac={robot.vacuum:.1f}")
        print(f"    block=({block.x:.3f},{block.y:.3f}) gripper=({gx:.3f},{gy:.3f}) phase={pick._phase}")
else:
    print(f"  Pick FAILED after 200 steps")
    robot = extract_robot(obs)
    block = extract_rect(obs, block_name)
    gx, gy = get_gripper_pos(robot)
    print(f"  Final: robot=({robot.x:.3f},{robot.y:.3f}) arm={robot.arm_joint:.3f} theta={robot.theta:.3f} vac={robot.vacuum:.1f}")
    print(f"  block=({block.x:.3f},{block.y:.3f}) gripper=({gx:.3f},{gy:.3f})")
