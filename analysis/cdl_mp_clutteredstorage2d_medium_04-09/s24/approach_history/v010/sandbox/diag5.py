"""Test if held block can push existing shelf blocks upward."""
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

# SCENARIO A: Place block2 first (no block1 in shelf yet), then block1
print("=== SCENARIO A: Place block2 FIRST, then block1 ===")
env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, info = env.reset(seed=0)

print("Initial:", {n: f"({get_block_center(obs,n)[0]:.2f},{get_block_center(obs,n)[1]:.2f})" for n in BLOCK_NAMES})

# Pickup block2 first
pb2 = PickupBlock('block2', PRIMITIVES)
pb2.reset(obs)
for _ in range(500):
    if pb2.terminated(obs): break
    obs,_,t,tr,_ = env.step(pb2.step(obs))
    if t or tr: break

robot = extract_robot(obs)
print(f"Holding block2: vac={robot.vacuum:.1f}")

# Place block2
place = PlaceBlock(PRIMITIVES)
place.reset(obs)
for _ in range(500):
    if place.terminated(obs): break
    obs,_,t,tr,_ = env.step(place.step(obs))
    if t or tr: break

print("After placing block2:", {n: f"({get_block_center(obs,n)[0]:.2f},{get_block_center(obs,n)[1]:.2f}) {'IN' if is_block_in_shelf(obs,n) else 'OUT'}" for n in BLOCK_NAMES})

# Now try to place block1 and see if it can fit
pb1 = PickupBlock('block1', PRIMITIVES)
pb1.reset(obs)
for _ in range(500):
    if pb1.terminated(obs): break
    obs,_,t,tr,_ = env.step(pb1.step(obs))
    if t or tr: break

# Navigate to shelf position
for _ in range(300):
    r = extract_robot(obs)
    dx=0.236-r.x; dy=2.100-r.y
    dth=(math.pi/2)-r.theta
    while dth>math.pi: dth-=2*math.pi
    while dth<-math.pi: dth+=2*math.pi
    if abs(dx)<0.01 and abs(dy)<0.01 and abs(dth)<0.02: break
    obs,_,_,_,_ = env.step(np.array([np.clip(dx,-0.05,0.05),np.clip(dy,-0.05,0.05),np.clip(dth,-math.pi/16,math.pi/16),0.0,1.0],dtype=np.float32))

r = extract_robot(obs)
print(f"\nAt placement: robot=({r.x:.3f},{r.y:.3f}) arm={r.arm_joint:.3f}")
for name in BLOCK_NAMES:
    cx,cy = get_block_center(obs,name)
    print(f"  {name}: ({cx:.3f},{cy:.3f}) {'IN' if is_block_in_shelf(obs,name) else 'OUT'}")

# Try extending arm - does held block1 push block2 up?
print("\nExtending arm (holding block1) - will it push block2?")
prev = r.arm_joint
for step in range(100):
    r = extract_robot(obs)
    arm = r.arm_joint
    sy = r.y + math.sin(r.theta)*(arm+0.03)
    b0 = get_block_center(obs,'block0'); b1 = get_block_center(obs,'block1'); b2 = get_block_center(obs,'block2')
    if step % 5 == 0:
        print(f"  step={step}: arm={arm:.4f} suc_y={sy:.4f} b0y={b0[1]:.4f} b1y={b1[1]:.4f} b2y={b2[1]:.4f}")
    if arm >= 0.600:
        print(f"  SUCCESS: Reached arm={arm:.4f}")
        break
    if step > 10 and abs(arm - prev) < 0.001:
        print(f"  STUCK at arm={arm:.4f}")
        break
    prev = arm
    obs,_,_,_,_ = env.step(np.array([0.,0.,0.,0.02,1.0],dtype=np.float32))

print("\nFinal state:")
for name in BLOCK_NAMES:
    cx,cy = get_block_center(obs,name)
    print(f"  {name}: ({cx:.3f},{cy:.3f}) {'IN' if is_block_in_shelf(obs,name) else 'OUT'}")

# SCENARIO B: try extending arm with VAC=0 while positioned near shelf, no block held
print("\n=== SCENARIO B: push blocks with arm (no held block) but WITH force ===")
env2 = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs2, _ = env2.reset(seed=0)

# Place block1
pb = PickupBlock('block1', PRIMITIVES)
pb.reset(obs2)
for _ in range(500):
    if pb.terminated(obs2): break
    obs2,_,_,_,_ = env2.step(pb.step(obs2))
pl = PlaceBlock(PRIMITIVES)
pl.reset(obs2)
for _ in range(500):
    if pl.terminated(obs2): break
    obs2,_,_,_,_ = env2.step(pl.step(obs2))

# Navigate
for _ in range(300):
    r = extract_robot(obs2)
    dx=0.236-r.x; dy=2.100-r.y
    dth=(math.pi/2)-r.theta
    while dth>math.pi: dth-=2*math.pi
    while dth<-math.pi: dth+=2*math.pi
    if abs(dx)<0.01 and abs(dy)<0.01 and abs(dth)<0.02: break
    obs2,_,_,_,_ = env2.step(np.array([np.clip(dx,-0.05,0.05),np.clip(dy,-0.05,0.05),np.clip(dth,-math.pi/16,math.pi/16),0.0,0.0],dtype=np.float32))

# Try extending arm with VAC=0 - does physics allow any block movement?
print("\nExtending arm with VAC=1 but NO block (robot not holding):")
prev = extract_robot(obs2).arm_joint
for step in range(100):
    r = extract_robot(obs2)
    arm = r.arm_joint
    b1 = get_block_center(obs2,'block1'); b0 = get_block_center(obs2,'block0')
    if step % 5 == 0:
        print(f"  step={step}: arm={arm:.4f} b0y={b0[1]:.4f} b1y={b1[1]:.4f}")
    if arm >= 0.700: print(f"  Reached!"); break
    if step > 10 and abs(arm-prev) < 0.001: print(f"  STUCK at {arm:.4f}"); break
    prev = arm
    obs2,_,_,_,_ = env2.step(np.array([0.,0.,0.,0.05,1.0],dtype=np.float32))  # try max force

print("\nFinal shelf state:")
for name in BLOCK_NAMES:
    cx,cy = get_block_center(obs2,name)
    print(f"  {name}: ({cx:.3f},{cy:.3f}) {'IN' if is_block_in_shelf(obs2,name) else 'OUT'}")
