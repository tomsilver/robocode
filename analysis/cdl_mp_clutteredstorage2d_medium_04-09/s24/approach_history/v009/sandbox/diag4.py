"""Test: can arm extend past block0 during first PlaceBlock?"""
import sys, os
sys.path.insert(0, '/sandbox')
os.chdir('/sandbox')
import numpy as np, math
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import (extract_robot, get_block_center, get_shelf_slot,
                          is_block_in_shelf, BLOCK_NAMES)
from primitives.motion_planning import BiRRT
from behaviors import PickupBlock
PRIMITIVES = {'BiRRT': BiRRT}

env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, info = env.reset(seed=0)

# Pickup block1
pb = PickupBlock('block1', PRIMITIVES)
pb.reset(obs)
for _ in range(500):
    if pb.terminated(obs): break
    obs,_,t,tr,_ = env.step(pb.step(obs))
    if t or tr: break

robot = extract_robot(obs)
print(f"Holding block1: robot=({robot.x:.3f},{robot.y:.3f}) arm={robot.arm_joint:.3f} vac={robot.vacuum:.1f}")
for name in BLOCK_NAMES:
    cx, cy = get_block_center(obs, name)
    print(f"  {name}: ({cx:.3f},{cy:.3f})")

# Navigate to placement position
for _ in range(300):
    robot = extract_robot(obs)
    dx=0.236-robot.x; dy=2.100-robot.y
    dth=(math.pi/2)-robot.theta
    while dth>math.pi: dth-=2*math.pi
    while dth<-math.pi: dth+=2*math.pi
    if abs(dx)<0.01 and abs(dy)<0.01 and abs(dth)<0.02: break
    obs,_,_,_,_ = env.step(np.array([np.clip(dx,-0.05,0.05),np.clip(dy,-0.05,0.05),np.clip(dth,-math.pi/16,math.pi/16),0.0,1.0],dtype=np.float32))

robot = extract_robot(obs)
print(f"\nAt placement position: robot=({robot.x:.3f},{robot.y:.3f}) arm={robot.arm_joint:.3f}")

# Test 1: Extend to arm=0.700 with VAC=1 (holding block1)
print("\nTest: Extend to 0.700 with VAC=1 (holding block1, only block0 in shelf):")
prev = robot.arm_joint
for step in range(200):
    robot = extract_robot(obs)
    arm = robot.arm_joint
    sy = robot.y + math.sin(robot.theta)*(arm+0.03)
    b0 = get_block_center(obs, 'block0')
    b1 = get_block_center(obs, 'block1')
    if step % 5 == 0 or arm < prev - 0.002:
        print(f"  step={step}: arm={arm:.4f} suc_y={sy:.4f} b0y={b0[1]:.4f} b1y={b1[1]:.4f}")
    if arm >= 0.700:
        print(f"  SUCCESS: Reached arm={arm:.4f}")
        break
    if step > 10 and abs(arm - prev) < 0.001:
        print(f"  STUCK at arm={arm:.4f}")
        break
    prev = arm
    obs,_,_,_,_ = env.step(np.array([0.,0.,0.,0.02,1.0],dtype=np.float32))

print("\nFinal state:")
for name in BLOCK_NAMES:
    cx, cy = get_block_center(obs, name)
    print(f"  {name}: ({cx:.3f},{cy:.3f}) in_shelf={is_block_in_shelf(obs, name)}")

# Test 2: Release block1 and check where it ended up
print("\nReleasing block1...")
for _ in range(5):
    obs,_,_,_,_ = env.step(np.array([0.,0.,0.,0.,0.0],dtype=np.float32))
print("After release:")
for name in BLOCK_NAMES:
    cx, cy = get_block_center(obs, name)
    print(f"  {name}: ({cx:.3f},{cy:.3f}) in_shelf={is_block_in_shelf(obs, name)}")

# Test 3: Now try to extend arm again (for block2 placement)
print("\nRetract arm to 0.200...")
for _ in range(50):
    robot = extract_robot(obs)
    if robot.arm_joint <= 0.205: break
    obs,_,_,_,_ = env.step(np.array([0.,0.,0.,-0.1,0.0],dtype=np.float32))

print("Now extend arm (for block2, no block2 held):")
prev = extract_robot(obs).arm_joint
for step in range(200):
    robot = extract_robot(obs)
    arm = robot.arm_joint
    sy = robot.y + math.sin(robot.theta)*(arm+0.03)
    if step % 5 == 0:
        b0 = get_block_center(obs, 'block0')
        b1 = get_block_center(obs, 'block1')
        print(f"  step={step}: arm={arm:.4f} suc_y={sy:.4f} b0y={b0[1]:.4f} b1y={b1[1]:.4f}")
    if arm >= 0.700:
        print(f"  SUCCESS: Reached arm={arm:.4f}")
        break
    if step > 10 and abs(arm - prev) < 0.001:
        print(f"  STUCK at arm={arm:.4f}")
        break
    prev = arm
    obs,_,_,_,_ = env.step(np.array([0.,0.,0.,0.02,0.0],dtype=np.float32))
