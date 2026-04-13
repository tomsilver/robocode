import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_rect, block_center

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

# Run until TempDropBehavior NAVIGATE starts
step = 0
while type(approach._current).__name__ != 'TempDropBehavior' or approach._current._phase != 'NAVIGATE':
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    if step > 200: break

print(f"TempDropBehavior NAVIGATE starts at step {step}")
robot = extract_robot(obs)
b0 = extract_rect(obs, 'block0')
b0cx, b0cy = block_center(b0)
print(f"  robot=({robot.x:.3f},{robot.y:.3f}) arm={robot.arm_joint:.3f}")
print(f"  block0 center=({b0cx:.3f},{b0cy:.3f}) theta={b0.theta:.3f}")
print(f"  TempDrop target=({approach._current._target_robot_x:.3f},{approach._current._target_robot_y:.3f})")
print(f"  Path length: {len(approach._current._path)} waypoints")

# Track rejected steps
rejected = 0
accepted = 0
for i in range(200):
    action = approach.get_action(obs)
    robot_before = extract_robot(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    robot_after = extract_robot(obs)
    dx = abs(robot_after.x - robot_before.x)
    dy = abs(robot_after.y - robot_before.y)
    if dx < 0.001 and dy < 0.001 and abs(action[0]) > 0.001:
        rejected += 1
    else:
        accepted += 1
    step += 1

print(f"After 200 steps: accepted={accepted}, rejected={rejected}")
robot = extract_robot(obs)
b0 = extract_rect(obs, 'block0')
b0cx, b0cy = block_center(b0)
print(f"  robot=({robot.x:.3f},{robot.y:.3f})")
print(f"  block0 center=({b0cx:.3f},{b0cy:.3f})")
