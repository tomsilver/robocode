"""Test: remove block0, then stack block1+block2 via pushing. Then add block0 back."""
import sys, os; sys.path.insert(0, '/sandbox'); os.chdir('/sandbox')
import numpy as np, math
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, get_block_center, is_block_in_shelf, BLOCK_NAMES, get_shelf_slot
PRIMITIVES = {}

def nav(env, obs, tx, ty, tth, vac=0., steps=400):
    for _ in range(steps):
        r=extract_robot(obs); dx=tx-r.x; dy=ty-r.y
        dth=tth-r.theta
        while dth>math.pi: dth-=2*math.pi
        while dth<-math.pi: dth+=2*math.pi
        if abs(dx)<0.015 and abs(dy)<0.015 and abs(dth)<0.03: break
        obs,_,_,_,_=env.step(np.array([np.clip(dx,-0.05,0.05),np.clip(dy,-0.05,0.05),np.clip(dth,-math.pi/16,math.pi/16),0.,vac],dtype=np.float32))
    return obs

def ext(env, obs, target, vac, steps=300):
    prev=extract_robot(obs).arm_joint; sc=0
    for s in range(steps):
        r=extract_robot(obs); a=r.arm_joint
        if a>=target-0.005: print(f"  arm={a:.4f} reached"); return obs
        if s>10 and abs(a-prev)<0.001:
            sc+=1
            if sc>15: print(f"  arm={a:.4f} STUCK"); return obs
        else: sc=0
        prev=a
        obs,_,_,_,_=env.step(np.array([0.,0.,0.,0.02,vac],dtype=np.float32))
    return obs

def retract(env, obs, vac=0.):
    for _ in range(40):
        if extract_robot(obs).arm_joint<0.21: break
        obs,_,_,_,_=env.step(np.array([0.,0.,0.,-0.1,vac],dtype=np.float32))
    return obs

def rel(env, obs):
    for _ in range(5):
        obs,_,_,_,_=env.step(np.array([0.,0.,0.,0.,0.],dtype=np.float32))
    return obs

env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs,_=env.reset(seed=0)
print("Init:", {n:f"({get_block_center(obs,n)[0]:.3f},{get_block_center(obs,n)[1]:.3f})" for n in BLOCK_NAMES})

slot=get_shelf_slot(obs)
print(f"Slot: {slot}")

# === STEP 1: Pick up block0 from shelf (approach from below) ===
print("\n=== STEP 1: Navigate to (0.236,2.305) to approach block0 ===")
# block0 at y=2.705. Approach from below: robot_y=2.705-0.4=2.305 (within slot x=0.236 to pass through slot)
obs=nav(env,obs, 0.236, 2.305, math.pi/2, vac=0.)
r=extract_robot(obs)
print(f"Robot: ({r.x:.3f},{r.y:.3f}) theta={math.degrees(r.theta):.1f}")

# Extend arm toward block0 with VAC=1, target arm=0.360 (suction at 2.305+0.360+0.030=2.695 near block0_bottom=2.685)
print("Extending toward block0:")
obs=ext(env,obs, 0.380, 1.)
r=extract_robot(obs); b0=get_block_center(obs,'block0')
print(f"Arm={r.arm_joint:.4f} vac={r.vacuum:.1f} b0=({b0[0]:.3f},{b0[1]:.3f})")

# Check if block0 is being held (check if block near suction)
from obs_helpers import suction_center_pos
r=extract_robot(obs)
sx,sy=suction_center_pos(r)
cx,cy=get_block_center(obs,'block0')
print(f"Suction: ({sx:.3f},{sy:.3f}) block0: ({cx:.3f},{cy:.3f}) dist={math.sqrt((sx-cx)**2+(sy-cy)**2):.4f}")

# === STEP 2: Retract and move block0 to safe location ===
print("\nRetract and move block0 to floor (x=1.5, y=1.0):")
obs=retract(env,obs,vac=1.)
obs=nav(env,obs, 1.5, 1.0, math.pi/2, vac=1.)
obs=retract(env,obs,vac=0.)
obs=rel(env,obs)
print("After dropping block0:", {n:f"({get_block_center(obs,n)[0]:.3f},{get_block_center(obs,n)[1]:.3f})" for n in BLOCK_NAMES})

# === STEP 3: Pick up block1 from floor and place in shelf ===
print("\n=== STEP 3: Pickup block1 ===")
from behaviors import PickupBlock
pb1=PickupBlock('block1', {})
pb1.reset(obs)
for _ in range(600):
    if pb1.terminated(obs): break
    obs,_,_,_,_=env.step(pb1.step(obs))
r=extract_robot(obs); b1=get_block_center(obs,'block1')
print(f"Held block1: arm={r.arm_joint:.3f} b1=({b1[0]:.3f},{b1[1]:.3f})")

# Navigate to adj_x: block1 is at b1[0], slot_cx=0.236. adj_x = 0.236 - (b1[0]-rx)
# For theta=pi/2, block_offset_x = b1[0]-r.x
r=extract_robot(obs)
block_offset_x = b1[0]-r.x
adj_x = 0.236 - block_offset_x
adj_x = max(0.08, min(0.38, adj_x))
print(f"adj_x={adj_x:.3f} (block_offset_x={block_offset_x:.3f})")

obs=nav(env,obs, adj_x, 2.100, math.pi/2, vac=1.)
r=extract_robot(obs); b1=get_block_center(obs,'block1')
print(f"After nav: robot_x={r.x:.3f} b1_x={b1[0]:.3f}")

# Extend to arm=0.700
print("Extending arm (block1, target=0.700):")
obs=ext(env,obs, 0.700, 1.)
r=extract_robot(obs); b1=get_block_center(obs,'block1'); b0=get_block_center(obs,'block0')
print(f"arm={r.arm_joint:.4f} b1=({b1[0]:.3f},{b1[1]:.3f}) b0=({b0[0]:.3f},{b0[1]:.3f})")
obs=rel(env,obs)

print("After releasing block1:")
for n in BLOCK_NAMES:
    cx,cy=get_block_center(obs,n); print(f"  {n}: ({cx:.3f},{cy:.3f}) in_shelf={is_block_in_shelf(obs,n)}")

# === STEP 4: Pick up block2, place in shelf (will push block1 up) ===
print("\n=== STEP 4: Pickup block2 ===")
obs=retract(env,obs,vac=0.)
pb2=PickupBlock('block2', {})
pb2.reset(obs)
for _ in range(600):
    if pb2.terminated(obs): break
    obs,_,_,_,_=env.step(pb2.step(obs))
r=extract_robot(obs); b2=get_block_center(obs,'block2')
block_offset_x2=b2[0]-r.x
adj_x2=0.236-block_offset_x2
adj_x2=max(0.08,min(0.38,adj_x2))
obs=nav(env,obs,adj_x2,2.100,math.pi/2,vac=1.)

print("Extending arm (block2, target=0.700, will push block1 up):")
obs=ext(env,obs,0.700,1.)
r=extract_robot(obs)
print(f"arm={r.arm_joint:.4f}")
for n in BLOCK_NAMES:
    cx,cy=get_block_center(obs,n); print(f"  {n}: ({cx:.3f},{cy:.3f})")
obs=rel(env,obs)
print("After releasing block2:")
for n in BLOCK_NAMES:
    cx,cy=get_block_center(obs,n); print(f"  {n}: ({cx:.3f},{cy:.3f}) in_shelf={is_block_in_shelf(obs,n)}")
