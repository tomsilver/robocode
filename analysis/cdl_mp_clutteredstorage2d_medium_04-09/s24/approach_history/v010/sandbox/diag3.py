"""Check if blocks can be pushed upward by the arm."""
import sys, os
sys.path.insert(0, '/sandbox')
os.chdir('/sandbox')
import numpy as np, math
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import (extract_robot, get_block_center, get_shelf_slot,
                          is_block_in_shelf, BLOCK_NAMES)
from primitives.motion_planning import BiRRT
from behaviors import PickupBlock, PlaceBlock
PRIMITIVES = {'BiRRT': BiRRT}

env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, info = env.reset(seed=0)

# Place block1 in shelf
pb = PickupBlock('block1', PRIMITIVES)
pb.reset(obs)
for _ in range(500):
    if pb.terminated(obs): break
    obs,_,t,tr,_ = env.step(pb.step(obs));
    if t or tr: break
place = PlaceBlock(PRIMITIVES)
place.reset(obs)
for _ in range(500):
    if place.terminated(obs): break
    obs,_,t,tr,_ = env.step(place.step(obs));
    if t or tr: break

print("Initial shelf state after placing block1:")
for name in BLOCK_NAMES:
    cx, cy = get_block_center(obs, name)
    print(f"  {name}: ({cx:.3f},{cy:.3f})")

# Navigate robot to x=0.236, y=2.100, theta=pi/2
for _ in range(300):
    robot = extract_robot(obs)
    dx=0.236-robot.x; dy=2.100-robot.y
    dth=(math.pi/2)-robot.theta
    while dth>math.pi: dth-=2*math.pi
    while dth<-math.pi: dth+=2*math.pi
    if abs(dx)<0.01 and abs(dy)<0.01 and abs(dth)<0.02: break
    obs,_,_,_,_ = env.step(np.array([np.clip(dx,-0.05,0.05),np.clip(dy,-0.05,0.05),np.clip(dth,-math.pi/16,math.pi/16),0.0,0.0],dtype=np.float32))

# VERY slowly push arm - tiny 0.005 steps, vacuum OFF
print("\nPushing arm VERY slowly (0.005 steps, VAC=0):")
for step in range(120):
    robot = extract_robot(obs)
    arm = robot.arm_joint
    if step % 10 == 0:
        sy = robot.y + math.sin(robot.theta)*(arm+0.03)
        b0 = get_block_center(obs, 'block0')
        b1 = get_block_center(obs, 'block1')
        print(f"  step={step}: arm={arm:.4f} suc_y={sy:.4f} b0y={b0[1]:.4f} b1y={b1[1]:.4f}")
    if arm >= 0.650:
        print(f"  Reached arm={arm:.4f}")
        break
    obs,_,_,_,_ = env.step(np.array([0.,0.,0.,0.005,0.0],dtype=np.float32))

print("\nFinal state:")
for name in BLOCK_NAMES:
    cx, cy = get_block_center(obs, name)
    print(f"  {name}: ({cx:.3f},{cy:.3f}) in_shelf={is_block_in_shelf(obs, name)}")

# Now check: if we push arm from different x position (robot at x=0.100)
print("\n=== Test: robot at different x positions ===")
env2 = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs2, _ = env2.reset(seed=0)

# Place block1
pb2 = PickupBlock('block1', PRIMITIVES)
pb2.reset(obs2)
for _ in range(500):
    if pb2.terminated(obs2): break
    obs2,_,t,tr,_ = env2.step(pb2.step(obs2));
    if t or tr: break

# Place it at x=0.350 (right side)
place2 = PlaceBlock(PRIMITIVES)
# Override target x in place behavior
place2.reset(obs2)
for _ in range(500):
    if place2.terminated(obs2): break
    obs2,_,t,tr,_ = env2.step(place2.step(obs2));
    if t or tr: break

print("After placing block1 at default position:")
for name in BLOCK_NAMES:
    cx, cy = get_block_center(obs2, name)
    print(f"  {name}: ({cx:.3f},{cy:.3f})")

# Navigate to x=0.100 (further left)
for _ in range(300):
    robot = extract_robot(obs2)
    dx=0.100-robot.x; dy=2.100-robot.y
    dth=(math.pi/2)-robot.theta
    while dth>math.pi: dth-=2*math.pi
    while dth<-math.pi: dth+=2*math.pi
    if abs(dx)<0.01 and abs(dy)<0.01 and abs(dth)<0.02: break
    obs2,_,_,_,_ = env2.step(np.array([np.clip(dx,-0.05,0.05),np.clip(dy,-0.05,0.05),np.clip(dth,-math.pi/16,math.pi/16),0.0,0.0],dtype=np.float32))

# Extend arm - will it hit block1 at different x?
robot = extract_robot(obs2)
print(f"\nRobot at x={robot.x:.3f}. Extending arm:")
for step in range(100):
    robot = extract_robot(obs2)
    arm = robot.arm_joint
    sy = robot.y + math.sin(robot.theta)*(arm+0.03)
    sx = robot.x + math.cos(robot.theta)*(arm+0.03)
    if step % 5 == 0:
        b1 = get_block_center(obs2, 'block1')
        print(f"  step={step}: arm={arm:.4f} suction=({sx:.3f},{sy:.3f}) b1=({b1[0]:.3f},{b1[1]:.3f})")
    if arm >= 0.700:
        print(f"  Reached!")
        break
    prev = arm
    obs2,_,_,_,_ = env2.step(np.array([0.,0.,0.,0.02,0.0],dtype=np.float32))
    if step > 5 and abs(extract_robot(obs2).arm_joint - prev) < 0.001:
        print(f"  STUCK at arm={arm:.4f}")
        break

# Check is slot x range big enough to place block2 at x=0.100?
slot = get_shelf_slot(obs2)
print(f"\nSlot x=[{slot[0]:.3f},{slot[0]+slot[2]:.3f}] (block needs center in [{slot[0]+0.04:.3f},{slot[0]+slot[2]-0.04:.3f}])")
print(f"Robot at x=0.100 -> arm at x=0.100 -> block2 would be at x~0.100")
print(f"In slot? {slot[0]+0.04 < 0.100 < slot[0]+slot[2]-0.04}")
