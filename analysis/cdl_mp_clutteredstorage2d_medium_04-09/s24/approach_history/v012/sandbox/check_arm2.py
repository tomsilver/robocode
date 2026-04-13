import sys, os
sys.path.insert(0, '/sandbox')
os.chdir('/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import *
from act_helpers import make_action
import math

from approach import GeneratedApproach
from primitives.motion_planning import BiRRT
PRIMITIVES = {'BiRRT': BiRRT}

# Run until block1 placed and robot retracted
env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, _ = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, PRIMITIVES)
approach.reset(obs, {})

for step in range(200):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    r = extract_robot(obs)
    cur = approach._current
    if type(cur).__name__ == 'PickupBlock' and r.vacuum < 0.5 and step > 50:
        break

r = extract_robot(obs)
print(f"State: robot=({r.x:.3f},{r.y:.3f},{r.theta:.3f}) arm={r.arm_joint:.3f}")
for name in BLOCK_NAMES:
    cx, cy = get_block_center(obs, name)
    print(f"  {name}: ({cx:.3f},{cy:.3f}) in={is_block_in_shelf(obs, name)}")

# Now navigate robot to (0.222, 2.100) theta=pi/2 manually
print("\nNavigating to shelf position manually...")
target_x, target_y, target_th = 0.222, 2.100, math.pi/2
for step in range(200):
    r = extract_robot(obs)
    dx = target_x - r.x
    dy = target_y - r.y
    dth_err = ((target_th - r.theta + math.pi) % (2*math.pi)) - math.pi
    if abs(dx)<0.01 and abs(dy)<0.01 and abs(dth_err)<0.05:
        break
    adx = np.clip(dx, -0.05, 0.05)
    ady = np.clip(dy, -0.05, 0.05)
    dth = np.clip(dth_err, -math.pi/16, math.pi/16)
    obs, _, _, _, _ = env.step(make_action(adx, ady, dth, 0, 0))

r = extract_robot(obs)
print(f"Robot at: ({r.x:.3f},{r.y:.3f},{r.theta:.3f}) arm={r.arm_joint:.3f}")

# Now extend arm slowly
print("\nExtending arm from position near shelf:")
prev_arm = r.arm_joint
for step in range(50):
    obs, _, _, _, _ = env.step(make_action(0, 0, 0, 0.02, 1.0))
    r = extract_robot(obs)
    b1cx, b1cy = get_block_center(obs, 'block1')
    b0cx, b0cy = get_block_center(obs, 'block0')
    if step < 5 or step % 5 == 4 or abs(r.arm_joint - prev_arm) < 0.002:
        print(f"  s={step} arm={r.arm_joint:.3f} b1=({b1cx:.3f},{b1cy:.3f}) b0=({b0cx:.3f},{b0cy:.3f})")
    prev_arm = r.arm_joint

