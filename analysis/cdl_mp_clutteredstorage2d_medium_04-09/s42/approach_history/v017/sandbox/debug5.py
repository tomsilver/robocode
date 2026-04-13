"""Trace PlaceInShelf step by step after successful pick."""
import sys; sys.path.insert(0, '/sandbox')
import numpy as np, math
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
# Run pick until terminated
for _ in range(100):
    if pick.terminated(obs):
        break
    action = pick.step(obs)
    obs, _, _, _, _ = env.step(action)

robot = extract_robot(obs)
block = extract_rect(obs, block_name)
print(f"After pick: robot=({robot.x:.3f},{robot.y:.3f}) arm={robot.arm_joint:.3f} th={robot.theta:.3f} vac={robot.vacuum:.1f}")
print(f"  block_center={block_center(block)}")
place.reset(obs)
print(f"  arm_offset={place._arm_offset:.3f} perp_offset={place._perp_offset:.3f}")

prev_phase = -1
for step in range(200):
    robot = extract_robot(obs)
    block = extract_rect(obs, block_name)
    cx, cy = block_center(block)
    ph = place._phase

    if ph != prev_phase:
        print(f"\n--- Phase {ph} at step {step} ---")
        print(f"  robot=({robot.x:.4f},{robot.y:.4f}) arm={robot.arm_joint:.4f} th={robot.theta:.4f} vac={robot.vacuum:.1f}")
        print(f"  block=({cx:.4f},{cy:.4f}) in_shelf={is_block_in_shelf(obs, block_name)}")
        prev_phase = ph

    if place.terminated(obs):
        print(f"\n=== TERMINATED at step {step} ===")
        print(f"  block=({cx:.4f},{cy:.4f}) in_shelf={is_block_in_shelf(obs, block_name)}")
        break

    if step < 3 or step % 10 == 0:
        print(f"  step {step:3d}: ph={ph} arm={robot.arm_joint:.3f} th={robot.theta:.3f} bc=({cx:.3f},{cy:.3f}) q={len(place._actions)}")

    action = place.step(obs)
    obs, _, _, _, _ = env.step(action)
else:
    robot = extract_robot(obs)
    block = extract_rect(obs, block_name)
    cx, cy = block_center(block)
    print(f"\nFAILED. Final: robot=({robot.x:.3f},{robot.y:.3f}) arm={robot.arm_joint:.3f} th={robot.theta:.3f}")
    print(f"  block=({cx:.3f},{cy:.3f}) in_shelf={is_block_in_shelf(obs, block_name)}")
