"""Trace every single step of pick."""
import sys; sys.path.insert(0, '/sandbox')
import numpy as np, math
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from obs_helpers import extract_robot, extract_rect, get_gripper_pos, block_center
primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv("kinder/ClutteredStorage2D-b3-v0")
obs, info = env.reset(seed=0)

from behaviors import PickBlock
pick = PickBlock('block1', primitives)
pick.reset(obs)

print(f"Initial: phase={pick._phase} actions={len(pick._actions)}")

for step in range(30):
    robot = extract_robot(obs)
    block = extract_rect(obs, 'block1')
    gx, gy = get_gripper_pos(robot)
    cx, cy = block_center(block)
    dist = math.sqrt((gx-cx)**2+(gy-cy)**2)

    term = pick.terminated(obs)
    print(f"Step {step:2d}: robot=({robot.x:.3f},{robot.y:.3f}) arm={robot.arm_joint:.3f} th={robot.theta:.3f} vac={robot.vacuum:.0f} "
          f"gripper=({gx:.3f},{gy:.3f}) bc=({cx:.3f},{cy:.3f}) dist={dist:.3f} ph={pick._phase} q={len(pick._actions)} term={term}")

    if term:
        print("TERMINATED!")
        break

    action = pick.step(obs)
    obs, _, _, _, _ = env.step(action)
