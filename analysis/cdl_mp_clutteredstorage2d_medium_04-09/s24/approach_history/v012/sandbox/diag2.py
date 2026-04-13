"""Check shelf physical structure and test block stacking."""
import sys, os
sys.path.insert(0, '/sandbox')
os.chdir('/sandbox')
import numpy as np
import math
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import (extract_robot, extract_rect, get_block_center,
                          get_shelf_slot, is_block_in_shelf, BLOCK_NAMES,
                          SHELF_BASE_IDX, SHELF_FEATURES)

env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, info = env.reset(seed=0)

# Print raw shelf obs
print("=== SHELF RAW OBS ===")
base = SHELF_BASE_IDX
for i, fname in enumerate(SHELF_FEATURES):
    print(f"  [{base+i}] {fname} = {obs[base+i]:.4f}")

# Also check if blocks are static
from obs_helpers import BLOCK_BASE_INDICES, RECT_FEATURES
print("\n=== BLOCK STATIC VALUES ===")
for i, name in enumerate(BLOCK_NAMES):
    bidx = BLOCK_BASE_INDICES[i]
    static_idx = RECT_FEATURES.index('static')
    print(f"  {name} static={obs[bidx+static_idx]:.1f}")

# Test: try to extend arm WITHOUT vacuum past block1 level
# First manually place block1 in shelf
print("\n=== Test: extend arm with VAC=0 (no grab) ===")
from primitives.motion_planning import BiRRT
from behaviors import PickupBlock, PlaceBlock
PRIMITIVES = {'BiRRT': BiRRT}

pb = PickupBlock('block1', PRIMITIVES)
pb.reset(obs)
for _ in range(500):
    if pb.terminated(obs): break
    obs, _, t, tr, _ = env.step(pb.step(obs))
    if t or tr: break

place = PlaceBlock(PRIMITIVES)
place.reset(obs)
for _ in range(500):
    if place.terminated(obs): break
    obs, _, t, tr, _ = env.step(place.step(obs))
    if t or tr: break

print("Block1 placed. Shelf state:")
for name in BLOCK_NAMES:
    cx, cy = get_block_center(obs, name)
    r = extract_rect(obs, name)
    in_shelf = is_block_in_shelf(obs, name)
    cos_t = math.cos(r.theta); sin_t = math.sin(r.theta)
    hw = r.width/2; hh = r.height/2
    dx_half = abs(hw*cos_t) + abs(hh*sin_t)
    dy_half = abs(hw*sin_t) + abs(hh*cos_t)
    print(f"  {name}: center=({cx:.3f},{cy:.3f}) y=[{cy-dy_half:.3f},{cy+dy_half:.3f}] in_shelf={in_shelf}")

# Navigate robot to shelf position
print("\nNavigating to x=0.236, y=2.100, theta=pi/2...")
for _ in range(300):
    robot = extract_robot(obs)
    dx = 0.236-robot.x; dy = 2.100-robot.y
    dth = (math.pi/2) - robot.theta
    while dth>math.pi: dth-=2*math.pi
    while dth<-math.pi: dth+=2*math.pi
    if abs(dx)<0.01 and abs(dy)<0.01 and abs(dth)<0.02: break
    adx=np.clip(dx,-0.05,0.05); ady=np.clip(dy,-0.05,0.05); adth=np.clip(dth,-math.pi/16,math.pi/16)
    obs,_,_,_,_ = env.step(np.array([adx,ady,adth,0.0,0.0],dtype=np.float32))

robot = extract_robot(obs)
print(f"Robot: ({robot.x:.3f},{robot.y:.3f}) th={robot.theta:.3f} arm={robot.arm_joint:.3f}")

# Extend arm with VAC=0 to see if physics allows passing through block1
print("\nExtending arm VAC=0 (no vacuum):")
prev = robot.arm_joint
for step in range(100):
    robot = extract_robot(obs)
    target = 0.750
    darm = min(0.02, target - robot.arm_joint)
    if robot.arm_joint >= target-0.005:
        print(f"  Reached {robot.arm_joint:.4f} at step {step}")
        break
    if step > 5 and abs(robot.arm_joint - prev) < 0.001:
        print(f"  STUCK at {robot.arm_joint:.4f} (step {step})")
        break
    prev = robot.arm_joint
    if step % 5 == 0:
        sx = robot.x + math.cos(robot.theta)*(robot.arm_joint+0.03)
        sy = robot.y + math.sin(robot.theta)*(robot.arm_joint+0.03)
        print(f"  step={step}: arm={robot.arm_joint:.4f} suction_y={sy:.3f}")
    obs,_,_,_,_ = env.step(np.array([0.,0.,0.,darm,0.0],dtype=np.float32))

print("After arm extension (VAC=0):")
for name in BLOCK_NAMES:
    cx, cy = get_block_center(obs, name)
    in_shelf = is_block_in_shelf(obs, name)
    print(f"  {name}: center=({cx:.3f},{cy:.3f}) in_shelf={in_shelf}")

# Now try VAC=1 from current arm position
robot = extract_robot(obs)
print(f"\nNow with VAC=1 from arm={robot.arm_joint:.3f}, continuing extension:")
prev = robot.arm_joint
for step in range(60):
    robot = extract_robot(obs)
    target = 0.750
    darm = min(0.02, target - robot.arm_joint)
    if robot.arm_joint >= target-0.005:
        print(f"  Reached {robot.arm_joint:.4f} at step {step}")
        break
    if step > 5 and abs(robot.arm_joint - prev) < 0.001:
        print(f"  STUCK at {robot.arm_joint:.4f} (step {step}) with VAC=1")
        break
    prev = robot.arm_joint
    sx = robot.x + math.cos(robot.theta)*(robot.arm_joint+0.03)
    sy = robot.y + math.sin(robot.theta)*(robot.arm_joint+0.03)
    print(f"  step={step}: arm={robot.arm_joint:.4f} suction_y={sy:.3f}")
    obs,_,_,_,_ = env.step(np.array([0.,0.,0.,darm,1.0],dtype=np.float32))
