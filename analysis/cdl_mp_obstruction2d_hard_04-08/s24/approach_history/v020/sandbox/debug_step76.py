"""Debug what happens at step 76 - after first clear, second clear starts."""
import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, extract_surface, NUM_OBSTRUCTIONS

primitives = {"BiRRT": BiRRT}
env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")

obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

for step in range(100):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    r = extract_robot(obs)

    if step >= 74:
        cur = approach._current
        print(f"step={step+1} phase={cur._phase} robot=({r['x']:.3f},{r['y']:.3f}) theta={r['theta']:.3f} arm={r['arm_joint']:.3f} vac={r['vacuum']:.0f}")
        print(f"  action={action}")
        for i in range(NUM_OBSTRUCTIONS):
            ob = extract_obstruction(obs, i)
            print(f"  obs{i}: cx={ob['cx']:.3f} cy={ob['cy']:.3f} x1={ob['x1']:.3f} y1={ob['y1']:.3f} x2={ob['x2']:.3f} y2={ob['y2']:.3f}")

    if terminated or truncated:
        print(f"Done at step={step+1}")
        break

env.close()
