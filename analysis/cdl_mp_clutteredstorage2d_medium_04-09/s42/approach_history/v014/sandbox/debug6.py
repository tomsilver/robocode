"""Fine-grained trace of PlaceInShelf rotation issue."""
import sys; sys.path.insert(0, '/sandbox')
import numpy as np, math
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from obs_helpers import extract_robot, extract_rect, is_block_in_shelf, block_center
primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv("kinder/ClutteredStorage2D-b3-v0")
obs, info = env.reset(seed=0)

from behaviors import PickBlock, PlaceInShelf
pick = PickBlock('block1', primitives)
place = PlaceInShelf('block1', primitives, slot_index=0)

pick.reset(obs)
for _ in range(100):
    if pick.terminated(obs): break
    action = pick.step(obs)
    obs, _, _, _, _ = env.step(action)

place.reset(obs)
print(f"PlaceInShelf reset: arm_offset={place._arm_offset:.3f} perp={place._perp_offset:.3f}")

# Run and trace ALL steps
for step in range(120):
    robot = extract_robot(obs)
    block = extract_rect(obs, 'block1')
    cx, cy = block_center(block)
    print(f"s{step:3d} ph={place._phase} arm={robot.arm_joint:.3f} th={robot.theta:.4f} vac={robot.vacuum:.0f} bc=({cx:.3f},{cy:.3f}) q={len(place._actions)} in_shelf={is_block_in_shelf(obs,'block1')}")
    if place.terminated(obs):
        print("TERMINATED!")
        break
    action = place.step(obs)
    obs, _, _, _, _ = env.step(action)
