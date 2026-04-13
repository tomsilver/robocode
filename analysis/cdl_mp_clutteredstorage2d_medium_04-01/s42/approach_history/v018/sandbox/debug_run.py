"""Debug run - watch what happens step by step."""
import sys
sys.path.insert(0, '/sandbox')

import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT

from obs_helpers import (
    extract_robot, extract_block, extract_shelf_inner, block_center, block_vertices,
    is_block_in_shelf, is_holding_block, SHELF_FLOOR_Y
)
from behaviors import MoveBlock0UpBehavior, PickBlockBehavior, PlaceBlockBehavior

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv("kinder/ClutteredStorage2D-b3-v0")
obs, info = env.reset(seed=0)

# Print initial state
robot = extract_robot(obs)
shelf = extract_shelf_inner(obs)
print(f"Shelf inner: x=[{shelf.x:.3f},{shelf.x_max:.3f}], y=[{shelf.y:.3f},{shelf.y_max:.3f}], cx={shelf.cx:.3f}")
for i in range(3):
    b = extract_block(obs, i)
    bcy = block_center(b)[1]
    verts = block_vertices(b)
    miny = min(vy for _, vy in verts)
    maxy = max(vy for _, vy in verts)
    print(f"Block{i}: center=({block_center(b)[0]:.3f},{bcy:.3f}), y=[{miny:.4f},{maxy:.4f}], theta={b.theta:.3f}, in_shelf={is_block_in_shelf(obs, i)}")
print(f"Robot: ({robot.x:.3f},{robot.y:.3f}), theta={robot.theta:.3f}, arm={robot.arm_joint:.3f}")
print()

# Test MoveBlock0UpBehavior
mover = MoveBlock0UpBehavior(primitives)
print(f"MoveBlock0UpBehavior.initializable: {mover.initializable(obs)}")
mover.reset(obs)
print(f"Plan has {len(mover._actions)} actions")

# Run it step by step
N_DEBUG = 300
for step in range(N_DEBUG):
    done = mover.terminated(obs)
    if done:
        print(f"Mover terminated at step {step}!")
        break
    action = mover.step(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if step % 20 == 0:
        robot = extract_robot(obs)
        b0 = extract_block(obs, 0)
        bcy = block_center(b0)[1]
        verts = block_vertices(b0)
        miny = min(vy for _, vy in verts)
        print(f"  step={step}: robot=({robot.x:.3f},{robot.y:.3f}), arm={robot.arm_joint:.3f}, vac={robot.vacuum:.1f}, b0_cy={bcy:.4f}, b0_miny={miny:.4f}")

robot = extract_robot(obs)
b0 = extract_block(obs, 0)
bcy0 = block_center(b0)[1]
verts = block_vertices(b0)
miny0 = min(vy for _, vy in verts)
maxy0 = max(vy for _, vy in verts)
print(f"\nAfter mover: robot=({robot.x:.3f},{robot.y:.3f}), arm={robot.arm_joint:.3f}")
print(f"Block0: center_y={bcy0:.4f}, y=[{miny0:.4f},{maxy0:.4f}]")
print(f"Mover terminated: {mover.terminated(obs)}")

env.close()
