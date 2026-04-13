"""Diagnose block placement geometry."""
import sys, os
sys.path.insert(0, '/sandbox')
os.chdir('/sandbox')

import numpy as np
import math
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import (extract_robot, extract_rect, get_block_center,
                          get_shelf_slot, is_block_in_shelf, BLOCK_NAMES,
                          suction_center_pos)
from primitives.motion_planning import BiRRT
from behaviors import PickupBlock, PlaceBlock
from act_helpers import make_action, VAC_ON, VAC_OFF

PRIMITIVES = {'BiRRT': BiRRT}

env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, info = env.reset(seed=0)

print("=== INITIAL STATE ===")
robot = extract_robot(obs)
print(f"Robot: x={robot.x:.3f} y={robot.y:.3f} th={robot.theta:.3f} arm={robot.arm_joint:.3f} vac={robot.vacuum:.1f}")

slot = get_shelf_slot(obs)
print(f"Shelf slot: x1={slot[0]:.3f} y1={slot[1]:.3f} w={slot[2]:.3f} h={slot[3]:.3f}")
print(f"  -> x=[{slot[0]:.3f}, {slot[0]+slot[2]:.3f}] y=[{slot[1]:.3f}, {slot[1]+slot[3]:.3f}]")

for name in BLOCK_NAMES:
    r = extract_rect(obs, name)
    cx, cy = get_block_center(obs, name)
    in_shelf = is_block_in_shelf(obs, name)
    # Compute extents
    hw = r.width/2; hh = r.height/2; t = r.theta
    # corners
    c1x = r.x; c1y = r.y
    print(f"{name}: corner=({r.x:.3f},{r.y:.3f}) theta={r.theta:.3f} w={r.width:.3f} h={r.height:.3f}")
    print(f"  center=({cx:.3f},{cy:.3f}) in_shelf={in_shelf}")

# Place block1 manually - navigate to position, then extend arm with verbose output
print("\n=== SIMULATING MANUAL PLACE ===")
print("Placing robot at (0.236, 2.100) theta=pi/2 arm=0.200, then extending...")

# First: run PickupBlock for block1
print("\n-- Running PickupBlock(block1) --")
pb = PickupBlock('block1', PRIMITIVES)
pb.reset(obs)

for step in range(500):
    if pb.terminated(obs):
        print(f"  PickupBlock terminated at step {step}")
        break
    action = pb.step(obs)
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        break

robot = extract_robot(obs)
print(f"After PickupBlock: robot=({robot.x:.3f},{robot.y:.3f}) arm={robot.arm_joint:.3f} vac={robot.vacuum:.1f}")
for name in BLOCK_NAMES:
    cx, cy = get_block_center(obs, name)
    in_shelf = is_block_in_shelf(obs, name)
    print(f"  {name} center=({cx:.3f},{cy:.3f}) in_shelf={in_shelf}")

# Run PlaceBlock for block1
print("\n-- Running PlaceBlock (block1) --")
place = PlaceBlock(PRIMITIVES)
place.reset(obs)

for step in range(500):
    if place.terminated(obs):
        print(f"  PlaceBlock terminated at step {step}")
        break
    action = place.step(obs)
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        break

robot = extract_robot(obs)
print(f"After PlaceBlock: robot=({robot.x:.3f},{robot.y:.3f}) arm={robot.arm_joint:.3f} vac={robot.vacuum:.1f}")
for name in BLOCK_NAMES:
    cx, cy = get_block_center(obs, name)
    r = extract_rect(obs, name)
    in_shelf = is_block_in_shelf(obs, name)
    print(f"  {name}: center=({cx:.3f},{cy:.3f}) theta={r.theta:.3f} in_shelf={in_shelf}")

slot = get_shelf_slot(obs)
print(f"Shelf slot: x=[{slot[0]:.3f},{slot[0]+slot[2]:.3f}] y=[{slot[1]:.3f},{slot[1]+slot[3]:.3f}]")

# Now try to manually place block2 by extending arm
print("\n-- Manual arm extension test from current robot position --")
print("First navigate robot to (0.236, 2.100) with theta=pi/2...")

# navigate back to place position
robot = extract_robot(obs)
print(f"  Current robot: ({robot.x:.3f},{robot.y:.3f})")

print("\n-- Running PickupBlock(block2) --")
pb2 = PickupBlock('block2', PRIMITIVES)
pb2.reset(obs)

for step in range(500):
    if pb2.terminated(obs):
        print(f"  PickupBlock2 terminated at step {step}")
        break
    action = pb2.step(obs)
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        break

robot = extract_robot(obs)
print(f"After PickupBlock2: robot=({robot.x:.3f},{robot.y:.3f}) arm={robot.arm_joint:.3f} vac={robot.vacuum:.1f}")
for name in BLOCK_NAMES:
    cx, cy = get_block_center(obs, name)
    r = extract_rect(obs, name)
    in_shelf = is_block_in_shelf(obs, name)
    print(f"  {name}: center=({cx:.3f},{cy:.3f}) theta={r.theta:.3f} in_shelf={in_shelf}")

print("\n-- Manual EXTEND for block2 with verbose output --")
# First navigate+orient
for step in range(300):
    robot = extract_robot(obs)
    dx = 0.236 - robot.x
    dy = 2.100 - robot.y
    dth_err = (math.pi/2) - robot.theta
    while dth_err > math.pi: dth_err -= 2*math.pi
    while dth_err < -math.pi: dth_err += 2*math.pi

    if abs(dx) < 0.015 and abs(dy) < 0.015 and abs(dth_err) < 0.03:
        break
    adx = np.clip(dx, -0.05, 0.05)
    ady = np.clip(dy, -0.05, 0.05)
    adth = np.clip(dth_err, -math.pi/16, math.pi/16)
    obs, _, _, _, _ = env.step(np.array([adx, ady, adth, 0.0, 1.0], dtype=np.float32))

robot = extract_robot(obs)
print(f"After orient: robot=({robot.x:.3f},{robot.y:.3f}) th={robot.theta:.3f} arm={robot.arm_joint:.3f}")

# Get block positions
for name in BLOCK_NAMES:
    cx, cy = get_block_center(obs, name)
    r = extract_rect(obs, name)
    in_shelf = is_block_in_shelf(obs, name)
    print(f"  {name}: center=({cx:.3f},{cy:.3f}) theta={r.theta:.4f} in_shelf={in_shelf}")
    # Compute extents (approximate, axis-aligned approximation)
    hw = r.width/2; hh = r.height/2
    print(f"    Extents (approx): x_range=[{cx-max(hw,hh):.3f},{cx+max(hw,hh):.3f}]")
    # More precise: bounding box
    cos_t = math.cos(r.theta); sin_t = math.sin(r.theta)
    dx_half = abs(hw*cos_t) + abs(hh*sin_t)
    dy_half = abs(hw*sin_t) + abs(hh*cos_t)
    print(f"    BB: x=[{cx-dx_half:.3f},{cx+dx_half:.3f}] y=[{cy-dy_half:.3f},{cy+dy_half:.3f}]")

print("\nNow extending arm step by step:")
prev_arm = robot.arm_joint
for step in range(200):
    robot = extract_robot(obs)
    if robot.arm_joint - prev_arm < -0.005:
        print(f"  WARNING: arm retracted! {prev_arm:.4f} -> {robot.arm_joint:.4f}")
    prev_arm = robot.arm_joint

    target_arm = 0.700
    darm = min(0.02, max(-0.1, target_arm - robot.arm_joint))

    # Print every 10 steps
    if step % 5 == 0:
        sx = robot.x + math.cos(robot.theta) * (robot.arm_joint + 0.03)
        sy = robot.y + math.sin(robot.theta) * (robot.arm_joint + 0.03)
        bc2x, bc2y = get_block_center(obs, 'block2')
        print(f"  step={step}: arm={robot.arm_joint:.4f} suction=({sx:.3f},{sy:.3f}) block2=({bc2x:.3f},{bc2y:.3f}) darm_applied={darm:.4f}")

    if robot.arm_joint >= target_arm - 0.005:
        print(f"  Reached target arm {robot.arm_joint:.4f} at step {step}")
        break
    if abs(robot.arm_joint - prev_arm) < 0.001 and step > 20:
        print(f"  ARM STUCK at {robot.arm_joint:.4f} (step {step})")
        break

    obs, _, _, _, _ = env.step(np.array([0.0, 0.0, 0.0, darm, 1.0], dtype=np.float32))
    prev_arm = robot.arm_joint

robot = extract_robot(obs)
print(f"Final arm: {robot.arm_joint:.4f}")
for name in BLOCK_NAMES:
    cx, cy = get_block_center(obs, name)
    r = extract_rect(obs, name)
    in_shelf = is_block_in_shelf(obs, name)
    print(f"  {name}: center=({cx:.3f},{cy:.3f}) theta={r.theta:.4f} in_shelf={in_shelf}")
