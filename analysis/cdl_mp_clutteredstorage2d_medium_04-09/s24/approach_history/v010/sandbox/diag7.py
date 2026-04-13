"""Test: pickup block0 from shelf + held block pushing placed blocks."""
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

# ==== TEST 1: Pick up block0 from inside shelf ====
print("=== TEST 1: Pick up block0 from inside shelf ===")
env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, _ = env.reset(seed=0)

b0 = get_block_center(obs, 'block0')
slot = get_shelf_slot(obs)
print(f"Block0 initial: ({b0[0]:.3f},{b0[1]:.3f})")
print(f"Slot: x=[{slot[0]:.3f},{slot[0]+slot[2]:.3f}] y=[{slot[1]:.3f},{slot[1]+slot[3]:.3f}]")

# Navigate to x=0.236, y=2.100, theta=pi/2
for _ in range(300):
    r = extract_robot(obs)
    dx=0.236-r.x; dy=2.100-r.y
    dth=(math.pi/2)-r.theta
    while dth>math.pi: dth-=2*math.pi
    while dth<-math.pi: dth+=2*math.pi
    if abs(dx)<0.01 and abs(dy)<0.01 and abs(dth)<0.02: break
    obs,_,_,_,_ = env.step(np.array([np.clip(dx,-0.05,0.05),np.clip(dy,-0.05,0.05),np.clip(dth,-math.pi/16,math.pi/16),0.,0.],dtype=np.float32))

r = extract_robot(obs)
print(f"Robot at ({r.x:.3f},{r.y:.3f}) theta={math.degrees(r.theta):.1f}")

# Extend arm toward block0 with VAC=1
print("Extending arm with VAC=1 toward block0:")
prev = r.arm_joint
for step in range(150):
    r = extract_robot(obs)
    arm = r.arm_joint
    sy = r.y + math.sin(r.theta)*(arm+0.03)
    b0 = get_block_center(obs,'block0')
    if step%5==0:
        print(f"  step={step}: arm={arm:.4f} suction_y={sy:.4f} b0y={b0[1]:.4f} vac={r.vacuum:.1f}")
    if r.vacuum > 0.5:
        print(f"  GRABBED block0 at arm={arm:.4f}")
        break
    if step>5 and abs(arm-prev)<0.001:
        print(f"  ARM STUCK at {arm:.4f}")
        break
    prev=arm
    obs,_,_,_,_ = env.step(np.array([0.,0.,0.,0.02,1.],dtype=np.float32))

r = extract_robot(obs)
print(f"Final: arm={r.arm_joint:.4f} vac={r.vacuum:.1f}")
for n in BLOCK_NAMES:
    cx,cy=get_block_center(obs,n); print(f"  {n}: ({cx:.3f},{cy:.3f})")

# ==== TEST 2: Place block1 low, then try to push it up with held block2 ====
print("\n=== TEST 2: Place block1 at arm=0.500 (y~2.650), then extend arm further ===")
env2 = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs2, _ = env2.reset(seed=0)

# Place block1 at arm=0.500 (below block0)
pb1 = PickupBlock('block1', PRIMITIVES)
pb1.reset(obs2)
for _ in range(500):
    if pb1.terminated(obs2): break
    obs2,_,_,_,_ = env2.step(pb1.step(obs2))

# Navigate to placement position
for _ in range(300):
    r=extract_robot(obs2)
    dx=0.236-r.x; dy=2.100-r.y
    dth=(math.pi/2)-r.theta
    while dth>math.pi: dth-=2*math.pi
    while dth<-math.pi: dth+=2*math.pi
    if abs(dx)<0.01 and abs(dy)<0.01 and abs(dth)<0.02: break
    obs2,_,_,_,_ = env2.step(np.array([np.clip(dx,-0.05,0.05),np.clip(dy,-0.05,0.05),np.clip(dth,-math.pi/16,math.pi/16),0.,1.],dtype=np.float32))

# Extend to arm=0.500, release
print("Extending to arm=0.500 (holding block1):")
for step in range(100):
    r=extract_robot(obs2); arm=r.arm_joint
    if arm >= 0.498:
        print(f"  Releasing at arm={arm:.4f}")
        break
    if step>5 and abs(arm-(arm if step==0 else arm))<0.001: pass
    obs2,_,_,_,_ = env2.step(np.array([0.,0.,0.,0.02,1.],dtype=np.float32))

# Release
for _ in range(5):
    obs2,_,_,_,_ = env2.step(np.array([0.,0.,0.,0.,0.],dtype=np.float32))

r=extract_robot(obs2); b1=get_block_center(obs2,'block1'); b0=get_block_center(obs2,'block0')
print(f"After placing block1: b1=({b1[0]:.3f},{b1[1]:.3f}) b0=({b0[0]:.3f},{b0[1]:.3f})")

# Retract arm
for _ in range(30):
    r=extract_robot(obs2)
    if r.arm_joint < 0.21: break
    obs2,_,_,_,_ = env2.step(np.array([0.,0.,0.,-0.05,0.],dtype=np.float32))

# Now extend arm AGAIN (no block held) past block1 -- does block1 get pushed?
print("\nExtending empty arm (VAC=0) past block1 y-level:")
prev=extract_robot(obs2).arm_joint
for step in range(150):
    r=extract_robot(obs2); arm=r.arm_joint
    b1=get_block_center(obs2,'block1'); b0=get_block_center(obs2,'block0')
    if step%5==0:
        print(f"  step={step}: arm={arm:.4f} b1=({b1[0]:.3f},{b1[1]:.3f}) b0=({b0[0]:.3f},{b0[1]:.3f})")
    if arm>=0.700:
        print(f"  Reached arm={arm:.4f}")
        break
    if step>10 and abs(arm-prev)<0.001:
        print(f"  STUCK at arm={arm:.4f}")
        break
    prev=arm
    obs2,_,_,_,_ = env2.step(np.array([0.,0.,0.,0.02,0.],dtype=np.float32))

# ==== TEST 3: Test if robot can be at x=0.077 and extend arm past block1 (with VAC=0) ====
print("\n=== TEST 3: Robot at x=0.077, extend arm past block1 ===")
env3 = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs3, _ = env3.reset(seed=0)

# Place block1 at y=2.650
pb1b = PickupBlock('block1', PRIMITIVES)
pb1b.reset(obs3)
for _ in range(500):
    if pb1b.terminated(obs3): break
    obs3,_,_,_,_ = env3.step(pb1b.step(obs3))

# Navigate to x=0.236 for placement
for _ in range(300):
    r=extract_robot(obs3)
    dx=0.236-r.x; dy=2.100-r.y
    dth=(math.pi/2)-r.theta
    while dth>math.pi: dth-=2*math.pi
    while dth<-math.pi: dth+=2*math.pi
    if abs(dx)<0.01 and abs(dy)<0.01 and abs(dth)<0.02: break
    obs3,_,_,_,_ = env3.step(np.array([np.clip(dx,-0.05,0.05),np.clip(dy,-0.05,0.05),np.clip(dth,-math.pi/16,math.pi/16),0.,1.],dtype=np.float32))

for step in range(100):
    r=extract_robot(obs3); arm=r.arm_joint
    if arm>=0.498: break
    obs3,_,_,_,_ = env3.step(np.array([0.,0.,0.,0.02,1.],dtype=np.float32))
for _ in range(5):
    obs3,_,_,_,_ = env3.step(np.array([0.,0.,0.,0.,0.],dtype=np.float32))

b1=get_block_center(obs3,'block1')
print(f"Block1 placed at ({b1[0]:.3f},{b1[1]:.3f})")

# Retract and move to x=0.077
for _ in range(30):
    if extract_robot(obs3).arm_joint<0.21: break
    obs3,_,_,_,_ = env3.step(np.array([0.,0.,0.,-0.05,0.],dtype=np.float32))

for _ in range(300):
    r=extract_robot(obs3)
    dx=0.077-r.x; dy=2.100-r.y
    dth=(math.pi/2)-r.theta
    while dth>math.pi: dth-=2*math.pi
    while dth<-math.pi: dth+=2*math.pi
    if abs(dx)<0.01 and abs(dy)<0.01 and abs(dth)<0.02: break
    obs3,_,_,_,_ = env3.step(np.array([np.clip(dx,-0.05,0.05),np.clip(dy,-0.05,0.05),np.clip(dth,-math.pi/16,math.pi/16),0.,0.],dtype=np.float32))

r=extract_robot(obs3)
print(f"Robot now at ({r.x:.3f},{r.y:.3f})")

# Extend arm at x=0.077 past block1
print("Extending arm at x=0.077 past block1:")
prev=r.arm_joint
for step in range(150):
    r=extract_robot(obs3); arm=r.arm_joint
    b1=get_block_center(obs3,'block1'); b0=get_block_center(obs3,'block0')
    if step%5==0:
        print(f"  step={step}: arm={arm:.4f} b1=({b1[0]:.3f},{b1[1]:.3f}) b0=({b0[0]:.3f},{b0[1]:.3f})")
    if arm>=0.650:
        print(f"  SUCCESS: arm={arm:.4f}")
        break
    if step>10 and abs(arm-prev)<0.001:
        print(f"  STUCK at arm={arm:.4f}")
        break
    prev=arm
    obs3,_,_,_,_ = env3.step(np.array([0.,0.,0.,0.02,0.],dtype=np.float32))
