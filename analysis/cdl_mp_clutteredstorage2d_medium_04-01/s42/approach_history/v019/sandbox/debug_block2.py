"""Debug block 2 pick specifically."""
import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from obs_helpers import (
    extract_robot, extract_block, extract_shelf_inner, block_center, block_vertices,
    is_block_in_shelf, is_holding_block, gripper_tip_position, SHELF_FLOOR_Y
)
from behaviors import PickBlockBehavior, PlaceBlockBehavior, MoveBlock0UpBehavior, _block_face_approach

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv("kinder/ClutteredStorage2D-b3-v0")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
obs, info = env.reset(seed=seed)

# Fast forward to pick block2 (skip blocks 0 and 1 handling)
# 1. MoveBlock0Up
mover = MoveBlock0UpBehavior(primitives)
mover.reset(obs)
for _ in range(300):
    if mover.terminated(obs): break
    obs, _, _, _, _ = env.step(mover.step(obs))

# 2. Pick block1
pick1 = PickBlockBehavior(1, primitives)
pick1.reset(obs)
for _ in range(200):
    if pick1.terminated(obs): break
    obs, _, _, _, _ = env.step(pick1.step(obs))

print("After pick1:")
robot = extract_robot(obs)
b1 = extract_block(obs, 1)
print(f"  robot=({robot.x:.3f},{robot.y:.3f}), arm={robot.arm_joint:.3f}, vac={robot.vacuum:.1f}")
print(f"  block1: cy={block_center(b1)[1]:.4f}, held={is_holding_block(obs, 1)}")

# 3. Place block1
place1 = PlaceBlockBehavior(1, primitives)
place1.reset(obs)
for _ in range(200):
    if place1.terminated(obs): break
    obs, _, _, _, _ = env.step(place1.step(obs))

print("After place1:")
robot = extract_robot(obs)
b1 = extract_block(obs, 1)
b2 = extract_block(obs, 2)
print(f"  block1: cy={block_center(b1)[1]:.4f}, in_shelf={is_block_in_shelf(obs, 1)}")
print(f"  block2: cy={block_center(b2)[1]:.4f}")

# Now analyze block2 approach
b2 = extract_block(obs, 2)
robot = extract_robot(obs)
theta_approach, goal_x, goal_y, arm_desired = _block_face_approach(b2, robot.gripper_width)
print(f"\nBlock2 face approach:")
print(f"  block2: center={block_center(b2)}, theta={b2.theta:.3f}")
print(f"  theta_approach={theta_approach:.4f}, goal=({goal_x:.3f},{goal_y:.3f}), arm={arm_desired:.3f}")
print(f"  arm tip reach: arm_desired+gripper_width={arm_desired+robot.gripper_width:.4f}")

# 4. Pick block2 - trace carefully
pick2 = PickBlockBehavior(2, primitives)
pick2.reset(obs)
print(f"\nPick2 plan: {len(pick2._actions)} actions")

for step in range(250):
    done = pick2.terminated(obs)
    robot = extract_robot(obs)
    b2 = extract_block(obs, 2)
    tip = gripper_tip_position(robot)
    if step % 25 == 0:
        print(f"  step={step}: robot=({robot.x:.3f},{robot.y:.3f}), theta={robot.theta:.3f}, arm={robot.arm_joint:.3f}, vac={robot.vacuum:.1f}")
        print(f"    tip=({tip[0]:.3f},{tip[1]:.3f}), b2_cy={block_center(b2)[1]:.4f}, held={is_holding_block(obs, 2)}")
    if done:
        print(f"  Pick2 terminated at step {step}!")
        break
    action = pick2.step(obs)
    obs, _, _, _, _ = env.step(action)

robot = extract_robot(obs)
b2 = extract_block(obs, 2)
tip = gripper_tip_position(robot)
print(f"\nAfter pick2 loop:")
print(f"  robot=({robot.x:.3f},{robot.y:.3f}), theta={robot.theta:.3f}, arm={robot.arm_joint:.3f}, vac={robot.vacuum:.1f}")
print(f"  tip=({tip[0]:.3f},{tip[1]:.3f})")
print(f"  b2: center={block_center(b2)}, theta={b2.theta:.3f}")
b2verts = block_vertices(b2)
miny = min(vy for _, vy in b2verts)
maxy = max(vy for _, vy in b2verts)
print(f"  b2 y=[{miny:.4f},{maxy:.4f}]")
print(f"  held={is_holding_block(obs, 2)}")
dist_tip_to_b2 = np.hypot(tip[0]-block_center(b2)[0], tip[1]-block_center(b2)[1])
print(f"  dist tip to b2 center: {dist_tip_to_b2:.4f}")

env.close()
