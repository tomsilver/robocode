import sys; sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, NUM_OBSTRUCTIONS

primitives = {"BiRRT": BiRRT}
env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

for step in range(75):
    approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(approach.get_action(obs))

r = extract_robot(obs)
print(f"Robot: x={r['x']:.4f}, y={r['y']:.4f}")
for i in range(NUM_OBSTRUCTIONS):
    ob = extract_obstruction(obs, i)
    print(f"obs{i}: ({ob['x1']:.3f},{ob['y1']:.3f}) to ({ob['x2']:.3f},{ob['y2']:.3f}) cx={ob['cx']:.3f} cy={ob['cy']:.3f}")
env.close()
