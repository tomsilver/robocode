import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_rect, block_center, BLOCK_NAMES, is_block_in_shelf

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, info = env.reset(seed=1)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

prev = (type(approach._current).__name__, approach._current._phase)
for step in range(2000):
    cur = (type(approach._current).__name__, approach._current._phase)
    if cur != prev:
        robot = extract_robot(obs)
        in_s = [b for b in BLOCK_NAMES if is_block_in_shelf(obs, b)]
        print(f"step={step}: {prev[0]}.{prev[1]} -> {cur[0]}.{cur[1]} | xy=({robot.x:.2f},{robot.y:.2f}) arm={robot.arm_joint:.2f} vac={robot.vacuum:.0f} in={in_s}")
        prev = cur
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"DONE at step {step+1}, terminated={terminated}")
        break
robot = extract_robot(obs)
print(f"Final: xy=({robot.x:.2f},{robot.y:.2f}) arm={robot.arm_joint:.2f} phase={approach._current._phase}")
