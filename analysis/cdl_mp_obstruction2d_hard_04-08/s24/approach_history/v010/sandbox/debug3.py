import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction

primitives = {"BiRRT": BiRRT}
env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

for step in range(57):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)

r = extract_robot(obs)
print(f"Entering extend: robot=({r['x']:.4f},{r['y']:.4f}) theta={r['theta']:.4f} arm={r['arm_joint']:.4f}")
goal_x = approach._current._goal_x
goal_y = approach._current._goal_y
goal_theta = approach._current._goal_theta
print(f"  Goal: ({goal_x:.4f},{goal_y:.4f}) theta={goal_theta:.4f}")

for step in range(57, 80):
    r = extract_robot(obs)
    action = approach.get_action(obs)
    print(f"  step={step} pos=({r['x']:.4f},{r['y']:.4f}) arm={r['arm_joint']:.4f} vac={r['vacuum']:.0f} action=[{action[0]:.4f},{action[1]:.4f},{action[2]:.4f},{action[3]:.4f},{action[4]:.0f}]")
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
