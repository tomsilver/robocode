"""Debug full pick+place for one block."""
import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from obs_helpers import extract_robot, extract_rect, is_block_in_shelf, get_gripper_pos, block_center

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv("kinder/ClutteredStorage2D-b3-v0")
obs, info = env.reset(seed=0)

from behaviors import PickBlock, PlaceInShelf

block_name = 'block1'
pick = PickBlock(block_name, primitives)
place = PlaceInShelf(block_name, primitives, slot_index=0)

pick.reset(obs)
print(f"=== PickBlock({block_name}) ===")

for step in range(300):
    if pick.terminated(obs):
        robot = extract_robot(obs)
        block = extract_rect(obs, block_name)
        gx, gy = get_gripper_pos(robot)
        cx, cy = block_center(block)
        print(f"Pick terminated at step {step}: vac={robot.vacuum:.1f} gripper=({gx:.3f},{gy:.3f}) block_center=({cx:.3f},{cy:.3f})")
        break
    action = pick.step(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if step % 30 == 0:
        robot = extract_robot(obs)
        block = extract_rect(obs, block_name)
        gx, gy = get_gripper_pos(robot)
        cx, cy = block_center(block)
        print(f"  Step {step}: robot=({robot.x:.3f},{robot.y:.3f}) arm={robot.arm_joint:.3f} theta={robot.theta:.3f} vac={robot.vacuum:.1f}")
        print(f"    gripper=({gx:.3f},{gy:.3f}) block_center=({cx:.3f},{cy:.3f}) dist={((gx-cx)**2+(gy-cy)**2)**0.5:.3f} phase={pick._phase}")
else:
    print("Pick FAILED!")

print(f"\n=== PlaceInShelf({block_name}) ===")
print(f"PlaceInShelf initializable: {place.initializable(obs)}")
place.reset(obs)

for step in range(300):
    if place.terminated(obs):
        print(f"Place terminated at step {step}!")
        break
    action = place.step(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if step % 30 == 0:
        robot = extract_robot(obs)
        block = extract_rect(obs, block_name)
        gx, gy = get_gripper_pos(robot)
        cx, cy = block_center(block)
        print(f"  Step {step}: robot=({robot.x:.3f},{robot.y:.3f}) arm={robot.arm_joint:.3f} theta={robot.theta:.3f} vac={robot.vacuum:.1f}")
        print(f"    gripper=({gx:.3f},{gy:.3f}) block_center=({cx:.3f},{cy:.3f}) in_shelf={is_block_in_shelf(obs, block_name)} phase={place._phase}")
else:
    print("Place FAILED!")
    robot = extract_robot(obs)
    block = extract_rect(obs, block_name)
    gx, gy = get_gripper_pos(robot)
    cx, cy = block_center(block)
    print(f"  Final: robot=({robot.x:.3f},{robot.y:.3f}) arm={robot.arm_joint:.3f} theta={robot.theta:.3f} vac={robot.vacuum:.1f}")
    print(f"  gripper=({gx:.3f},{gy:.3f}) block_center=({cx:.3f},{cy:.3f}) in_shelf={is_block_in_shelf(obs, block_name)}")
