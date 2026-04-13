"""Test: pickup block0 from shelf, then stack block1+block2."""
import sys, os; sys.path.insert(0, '/sandbox'); os.chdir('/sandbox')
import numpy as np, math
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, get_block_center, is_block_in_shelf, BLOCK_NAMES
from primitives.motion_planning import BiRRT
from behaviors import PickupBlock, PlaceBlock
PRIMITIVES = {'BiRRT': BiRRT}

def nav(env, obs, tx, ty, tth, vac=0., steps=400):
    for _ in range(steps):
        r = extract_robot(obs)
        dx=tx-r.x; dy=ty-r.y
        dth=tth-r.theta
        while dth>math.pi: dth-=2*math.pi
        while dth<-math.pi: dth+=2*math.pi
        if abs(dx)<0.015 and abs(dy)<0.015 and abs(dth)<0.03: break
        obs,_,_,_,_ = env.step(np.array([np.clip(dx,-0.05,0.05),np.clip(dy,-0.05,0.05),np.clip(dth,-math.pi/16,math.pi/16),0.,vac],dtype=np.float32))
    return obs

def extend_arm(env, obs, target_arm, vac, steps=200):
    prev=extract_robot(obs).arm_joint; sc=0
    for step in range(steps):
        r=extract_robot(obs); arm=r.arm_joint
        if arm>=target_arm-0.005: print(f"  arm reached {arm:.4f}"); break
        if step>10 and abs(arm-prev)<0.001: sc+=1
        else: sc=0
        if sc>15: print(f"  arm STUCK at {arm:.4f}"); break
        prev=arm
        obs,_,_,_,_ = env.step(np.array([0.,0.,0.,0.02,vac],dtype=np.float32))
    return obs

def retract_arm(env, obs, vac=0., steps=50):
    for _ in range(steps):
        if extract_robot(obs).arm_joint < 0.21: break
        obs,_,_,_,_ = env.step(np.array([0.,0.,0.,-0.1,vac],dtype=np.float32))
    return obs

def release(env, obs, steps=5):
    for _ in range(steps):
        obs,_,_,_,_ = env.step(np.array([0.,0.,0.,0.,0.],dtype=np.float32))
    return obs

env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, _ = env.reset(seed=0)
print("Initial:", {n:f"({get_block_center(obs,n)[0]:.3f},{get_block_center(obs,n)[1]:.3f})" for n in BLOCK_NAMES})

# === STEP 1: Pick up block1 using PickupBlock ===
print("\n--- Step 1: PickupBlock(block1) ---")
pb = PickupBlock('block1', PRIMITIVES)
pb.reset(obs)
for _ in range(600):
    if pb.terminated(obs): break
    obs,_,t,tr,_ = env.step(pb.step(obs))
    if t or tr: break
r=extract_robot(obs); b1=get_block_center(obs,'block1')
print(f"After pickup: robot=({r.x:.3f},{r.y:.3f}) theta={math.degrees(r.theta):.1f} arm={r.arm_joint:.3f} vac={r.vacuum:.1f} b1=({b1[0]:.3f},{b1[1]:.3f})")

# === STEP 2: Navigate to shelf approach and extend carefully ===
print("\n--- Step 2: Navigate to shelf (x=0.236, y=2.100, theta=pi/2) ---")
obs = nav(env, obs, 0.236, 2.100, math.pi/2, vac=1.)
r=extract_robot(obs); b1=get_block_center(obs,'block1')
print(f"After nav: robot=({r.x:.3f},{r.y:.3f}) theta={math.degrees(r.theta):.1f} arm={r.arm_joint:.3f} b1=({b1[0]:.3f},{b1[1]:.3f}) theta_b1={math.degrees(0):.1f}")

# Extend arm slowly and watch what happens
print("Extending arm toward shelf (VAC=1, monitoring):")
prev=r.arm_joint
for step in range(200):
    r=extract_robot(obs); arm=r.arm_joint
    sy=r.y+math.sin(r.theta)*(arm+0.03)
    b0=get_block_center(obs,'block0'); b1=get_block_center(obs,'block1')
    if step%5==0:
        print(f"  step={step}: arm={arm:.4f} suc_y={sy:.4f} b0=({b0[0]:.3f},{b0[1]:.3f}) b1=({b1[0]:.3f},{b1[1]:.3f})")
    if arm>=0.550: print(f"  Reached arm={arm:.4f}"); break
    if step>10 and abs(arm-prev)<0.001: print(f"  STUCK at arm={arm:.4f}"); break
    prev=arm
    obs,_,_,_,_ = env.step(np.array([0.,0.,0.,0.02,1.],dtype=np.float32))

print("\nState at release:")
for n in BLOCK_NAMES:
    cx,cy=get_block_center(obs,n)
    print(f"  {n}: ({cx:.3f},{cy:.3f}) in_shelf={is_block_in_shelf(obs,n)}")

# Release
obs = release(env, obs)
print("After release:")
for n in BLOCK_NAMES:
    cx,cy=get_block_center(obs,n)
    print(f"  {n}: ({cx:.3f},{cy:.3f}) in_shelf={is_block_in_shelf(obs,n)}")
