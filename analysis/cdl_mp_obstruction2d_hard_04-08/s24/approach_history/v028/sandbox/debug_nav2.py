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

for step in range(140):
    approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(approach.get_action(obs))

print("After step 140:")
r = extract_robot(obs)
print(f"Robot: x={r['x']:.4f}, y={r['y']:.4f} theta={r['theta']:.4f}")
print(f"Phase: {approach._current._phase}, idx={approach._current._i}")
drop_x = approach._current._drop_x
drop_y = approach._current._drop_y
print(f"Drop target: ({drop_x:.4f}, {drop_y:.4f})")
for i in range(NUM_OBSTRUCTIONS):
    ob = extract_obstruction(obs, i)
    print(f"obs{i}: ({ob['x1']:.3f},{ob['y1']:.3f}) to ({ob['x2']:.3f},{ob['y2']:.3f}) static={ob['static']:.0f}")
env.close()
