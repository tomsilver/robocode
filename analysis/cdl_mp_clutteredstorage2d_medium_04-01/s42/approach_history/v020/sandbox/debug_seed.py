"""Debug pick for specific block."""
import sys
sys.path.insert(0, '/sandbox')
import numpy as np
import math
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from obs_helpers import (
    extract_robot, extract_block, extract_shelf_inner, block_center, block_vertices,
    is_block_in_shelf, is_holding_block, gripper_tip_position, SHELF_FLOOR_Y
)
from behaviors import MoveBlock0UpBehavior, PickBlockBehavior, _block_face_approach

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv("kinder/ClutteredStorage2D-b3-v0")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
bidx = int(sys.argv[2]) if len(sys.argv) > 2 else 1

obs, info = env.reset(seed=seed)

robot = extract_robot(obs)
for i in range(3):
    b = extract_block(obs, i)
    bcy = block_center(b)[1]
    bcx = block_center(b)[0]
    verts = block_vertices(b)
    miny = min(vy for _, vy in verts)
    maxy = max(vy for _, vy in verts)
    print(f"Block{i}: cx={bcx:.3f}, cy={bcy:.3f}, y=[{miny:.4f},{maxy:.4f}], theta={b.theta:.3f}, in_shelf={is_block_in_shelf(obs, i)}")

# Run MoveBlock0Up
mover = MoveBlock0UpBehavior(primitives)
mover.reset(obs)
for _ in range(300):
    if mover.terminated(obs): break
    obs, _, _, _, _ = env.step(mover.step(obs))

# Pick blocks before bidx
for prev_bidx in range(1, bidx):
    from behaviors import PlaceBlockBehavior
    pick = PickBlockBehavior(prev_bidx, primitives)
    pick.reset(obs)
    for _ in range(200):
        if pick.terminated(obs): break
        obs, _, _, _, _ = env.step(pick.step(obs))
    place = PlaceBlockBehavior(prev_bidx, primitives)
    place.reset(obs)
    for _ in range(200):
        if place.terminated(obs): break
        obs, _, _, _, _ = env.step(place.step(obs))
    print(f"\nAfter block{prev_bidx}: cy={block_center(extract_block(obs, prev_bidx))[1]:.4f}, in_shelf={is_block_in_shelf(obs, prev_bidx)}")

# Analyze target block approach
robot = extract_robot(obs)
block = extract_block(obs, bidx)
bcx, bcy = block_center(block)
verts = block_vertices(block)
print(f"\nTarget block{bidx}: cx={bcx:.3f}, cy={bcy:.3f}, theta={block.theta:.3f}")
print(f"  y=[{min(vy for _, vy in verts):.4f},{max(vy for _, vy in verts):.4f}]")
print(f"Robot at: ({robot.x:.3f},{robot.y:.3f})")

theta_approach, goal_x, goal_y, arm_desired = _block_face_approach(
    block, robot.gripper_width, robot_x=robot.x, robot_y=robot.y
)
print(f"Approach: theta={theta_approach:.4f}, goal=({goal_x:.3f},{goal_y:.3f}), arm={arm_desired:.3f}")

# Try pick
pick = PickBlockBehavior(bidx, primitives)
pick.reset(obs)
print(f"\nPick plan: {len(pick._actions)} actions")

for step in range(300):
    done = pick.terminated(obs)
    if done:
        robot = extract_robot(obs)
        block = extract_block(obs, bidx)
        tip = gripper_tip_position(robot)
        bcx, bcy = block_center(block)
        dist = math.hypot(tip[0]-bcx, tip[1]-bcy)
        print(f"Pick terminated at step={step}!")
        print(f"  robot=({robot.x:.3f},{robot.y:.3f}), theta={robot.theta:.3f}, arm={robot.arm_joint:.3f}, vac={robot.vacuum:.1f}")
        print(f"  tip=({tip[0]:.3f},{tip[1]:.3f}), b_cy={bcy:.4f}, dist_tip_to_center={dist:.4f}")
        break
    if step % 50 == 0:
        robot = extract_robot(obs)
        block = extract_block(obs, bidx)
        tip = gripper_tip_position(robot)
        bcx_now, bcy_now = block_center(block)
        print(f"  step={step}: robot=({robot.x:.3f},{robot.y:.3f}), arm={robot.arm_joint:.3f}, vac={robot.vacuum:.1f}, b_cy={bcy_now:.4f}")
    obs, _, _, _, _ = env.step(pick.step(obs))

env.close()
