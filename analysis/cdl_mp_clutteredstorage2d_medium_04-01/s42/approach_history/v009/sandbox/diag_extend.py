import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_rect, block_center, BLOCK_NAMES

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

# Fast-forward to PlaceBlockBehavior EXTEND
step = 0
while not (type(approach._current).__name__ == 'PlaceBlockBehavior' and approach._current._phase == 'EXTEND'):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    if step > 700: break

print(f"PlaceBlockBehavior EXTEND starts at step {step}")
robot = extract_robot(obs)
print(f"robot=({robot.x:.3f},{robot.y:.3f}) arm={robot.arm_joint:.3f} theta={robot.theta:.3f} vac={robot.vacuum}")
for bn in BLOCK_NAMES:
    rect = extract_rect(obs, bn)
    cx, cy = block_center(rect)
    print(f"  {bn}: center=({cx:.3f},{cy:.3f}) theta={rect.theta:.3f}")

# Run 100 steps and track arm and positions
for i in range(100):
    robot_prev = extract_robot(obs)
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    robot_now = extract_robot(obs)
    if abs(robot_now.arm_joint - robot_prev.arm_joint) < 0.001 and action[3] > 0.001:
        # Step rejected
        b1 = extract_rect(obs, 'block1')
        cx, cy = block_center(obs, 'block1') if False else block_center(b1)
        if i < 5:
            print(f"  REJECTED step {i}: arm={robot_now.arm_joint:.3f}, block1=({cx:.3f},{cy:.3f}) theta={b1.theta:.3f}")
    step += 1
print(f"After 100 steps: arm={extract_robot(obs).arm_joint:.3f}")
