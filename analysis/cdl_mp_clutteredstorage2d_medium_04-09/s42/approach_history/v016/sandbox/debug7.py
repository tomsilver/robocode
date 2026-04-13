"""Debug navigation issue: why doesn't robot reach (0.236, 2.050)?"""
import sys; sys.path.insert(0, '/sandbox')
import math
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from obs_helpers import (extract_robot, extract_rect, block_center,
                         SHELF_CENTER_X, SHELF_INNER_Y_MIN,
                         WORLD_MIN_X, WORLD_MAX_X, WORLD_MIN_Y, ROBOT_BASE_RADIUS)
from act_helpers import SHELF_APPROACH_Y, DX_LIM, path_to_actions
from behaviors import _make_birrt

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv("kinder/ClutteredStorage2D-b3-v0")
obs, info = env.reset(seed=0)

# Simulate end of pick: robot at (2.694, 1.394) theta=0 arm=0.300
# Then PlaceInShelf phase 0 retracts, phase 1 rotates → robot at ~(2.651, 1.406) theta=pi/2
# Let's check what BiRRT finds for navigation from there

from behaviors import PickBlock, PlaceInShelf
pick = PickBlock('block1', primitives)
place = PlaceInShelf('block1', primitives, slot_index=0)
pick.reset(obs)
for _ in range(100):
    if pick.terminated(obs): break
    obs, _, _, _, _ = env.step(pick.step(obs))

place.reset(obs)
# Run phase 0 (retract) and phase 1 (rotate)
for _ in range(20):
    if place._phase == 2: break
    action = place.step(obs)
    obs, _, _, _, _ = env.step(action)

robot = extract_robot(obs)
print(f"Before nav: robot=({robot.x:.4f},{robot.y:.4f}) th={robot.theta:.4f}")
goal_x = SHELF_CENTER_X + place._perp_offset
goal_y = SHELF_APPROACH_Y
print(f"Nav goal: ({goal_x:.4f}, {goal_y:.4f})")

# Test BiRRT
rng = np.random.default_rng(43)
birrt = _make_birrt(primitives, obs, ['block0','block2'], rng)
start = np.array([robot.x, robot.y])
goal = np.array([goal_x, goal_y])
path = birrt.query(start, goal)
if path is None:
    print("BiRRT failed! Using direct path.")
    path = [goal]
else:
    print(f"BiRRT success: {len(path)} waypoints")
    print(f"  First: {path[0]}")
    print(f"  Last: {path[-1]}")
    max_y = max(p[1] for p in path)
    print(f"  Max y in path: {max_y:.4f}")

# Count total actions
actions = path_to_actions(path, robot, 1.0)
print(f"Total actions: {len(actions)}")

# Now trace navigation step by step
for step, action in enumerate(actions):
    obs, _, _, _, _ = env.step(action)
    r = extract_robot(obs)
    if step % 10 == 0 or step == len(actions)-1:
        print(f"  nav step {step:3d}: robot=({r.x:.4f},{r.y:.4f})")
print(f"\nFinal robot pos: ({r.x:.4f},{r.y:.4f})")
print(f"Distance to goal: {math.sqrt((r.x-goal_x)**2+(r.y-goal_y)**2):.4f}")
