"""Test angled arm approach to place block ABOVE block0."""
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

# Navigate to angled position: robot at x=0.700, y=2.100, theta=2pi/3 (120deg)
target_x = 0.700
target_y = 2.100
target_theta = 2*math.pi/3  # 120 degrees

print(f"Target: ({target_x:.3f},{target_y:.3f}) theta={math.degrees(target_theta):.1f}°")

for step in range(500):
    r = extract_robot(obs)
    dx = target_x - r.x; dy = target_y - r.y
    dth = target_theta - r.theta
    while dth > math.pi: dth -= 2*math.pi
    while dth < -math.pi: dth += 2*math.pi
    if abs(dx)<0.015 and abs(dy)<0.015 and abs(dth)<0.03:
        break
    adx=np.clip(dx,-0.05,0.05); ady=np.clip(dy,-0.05,0.05)
    adth=np.clip(dth,-math.pi/16,math.pi/16)
    obs,_,t,tr,_ = env.step(np.array([adx,ady,adth,0.0,1.0],dtype=np.float32))
    if t or tr: break

r = extract_robot(obs)
print(f"At position: ({r.x:.3f},{r.y:.3f}) theta={math.degrees(r.theta):.1f}° arm={r.arm_joint:.3f}")
for name in BLOCK_NAMES:
    cx,cy = get_block_center(obs,name)
    print(f"  {name}: ({cx:.3f},{cy:.3f}) {'IN' if is_block_in_shelf(obs,name) else 'OUT'}")

# Now extend arm with VAC=1 (holding block1)
print("\nExtending arm (theta=120°, VAC=1):")
prev = r.arm_joint
for step in range(200):
    r = extract_robot(obs)
    arm = r.arm_joint
    sx = r.x + math.cos(r.theta)*(arm+0.03)
    sy = r.y + math.sin(r.theta)*(arm+0.03)
    b1 = get_block_center(obs,'block1')
    b0 = get_block_center(obs,'block0')

    in_shelf_b1 = is_block_in_shelf(obs,'block1')
    if step % 5 == 0:
        print(f"  step={step}: arm={arm:.4f} suction=({sx:.3f},{sy:.3f}) b1=({b1[0]:.3f},{b1[1]:.3f}) b0=({b0[0]:.3f},{b0[1]:.3f}) b1_in={in_shelf_b1}")

    if arm >= 0.750:
        print(f"  SUCCESS: Reached arm={arm:.4f}")
        break
    if step > 10 and abs(arm-prev) < 0.001:
        print(f"  STUCK at arm={arm:.4f}")
        break
    prev = arm
    obs,_,_,_,_ = env.step(np.array([0.,0.,0.,0.02,1.0],dtype=np.float32))

print("\nAfter arm extension:")
for name in BLOCK_NAMES:
    cx,cy = get_block_center(obs,name)
    print(f"  {name}: ({cx:.3f},{cy:.3f}) in_shelf={is_block_in_shelf(obs,name)}")

# Release and retract
print("\nRelease vacuum...")
for _ in range(5):
    obs,_,_,_,_ = env.step(np.array([0.,0.,0.,0.,0.0],dtype=np.float32))

print("After release:")
for name in BLOCK_NAMES:
    cx,cy = get_block_center(obs,name)
    print(f"  {name}: ({cx:.3f},{cy:.3f}) in_shelf={is_block_in_shelf(obs,name)}")

# Also check: what's the suction position at arm=0.515 (for below-block0 placement)
print("\n=== Also test: different x positions for theta=pi/2 ===")
for test_rx in [0.150, 0.200, 0.236, 0.300, 0.350]:
    # At theta=pi/2, arm=0.515:
    arm = 0.515
    sx = test_rx
    sy = 2.100 + arm + 0.030
    in_x = 0.117 < sx < 0.355
    print(f"  robot_x={test_rx:.3f}: suction=({sx:.3f},{sy:.3f}) in_shelf_x={in_x} arm={arm:.3f}")
    # Check gripper collision with block0 at this arm (block0: x=[0.095,0.377] y=[2.679,2.731])
    gripper_x_min = test_rx - 0.07
    gripper_x_max = test_rx + 0.07
    # Gripper y at arm: from arm_joint to arm_joint+0.02 roughly, extending up
    # At arm=0.515, gripper CENTER y ≈ 2.100+0.515+0.01=2.625; gripper extends slightly
    # But above block0? block0 is at y=[2.679+]
    # Gripper y range when arm=0.515: roughly [2.615, 2.655] - below block0 bottom 2.679. OK!
    print(f"    Gripper x=[{gripper_x_min:.3f},{gripper_x_max:.3f}] - overlap with block0 x=[0.095,0.377]: {max(0,min(gripper_x_max,0.377)-max(gripper_x_min,0.095)):.3f}")
