"""Trace every step of pick."""
import sys
sys.path.insert(0, '/sandbox')
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

prev_phase = -1
for step in range(35):
    robot = extract_robot(obs)
    block = extract_rect(obs, 'block1')
    gx, gy = get_gripper_pos(robot)
    cx, cy = block_center(block)
    if pick._phase != prev_phase:
        print(f"\n--- Phase {pick._phase} at step {step} ---")
        print(f"  robot=({robot.x:.4f},{robot.y:.4f}) theta={robot.theta:.4f} arm={robot.arm_joint:.4f} vac={robot.vacuum:.1f}")
        print(f"  gripper=({gx:.4f},{gy:.4f}) block_center=({cx:.4f},{cy:.4f}) dist={math.sqrt((gx-cx)**2+(gy-cy)**2):.4f}")
        print(f"  actions remaining: {len(pick._actions)}")
        prev_phase = pick._phase

    if pick.terminated(obs):
        print(f"\n=== TERMINATED at step {step} ===")
        print(f"  robot=({robot.x:.4f},{robot.y:.4f}) arm={robot.arm_joint:.4f} vac={robot.vacuum:.1f}")
        print(f"  gripper=({gx:.4f},{gy:.4f}) block_center=({cx:.4f},{cy:.4f})")
        break

    action = pick.step(obs)
    obs, _, _, _, _ = env.step(action)
