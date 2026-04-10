"""Trace obs3 nav_drop specifically."""
import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, extract_block, extract_surface, NUM_OBSTRUCTIONS

primitives = {"BiRRT": BiRRT}
env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

for step in range(200):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"DONE at step={step+1}")
        break

# Now trace carefully from step 144+
print("=== Starting detailed trace ===")
for step in range(200, 250):
    cur = approach._current
    r = extract_robot(obs)
    print(f"step={step} {type(cur).__name__}.{cur._phase} robot=({r['x']:.3f},{r['y']:.3f}) theta={r['theta']:.3f} arm={r['arm_joint']:.3f} vac={r['vacuum']:.0f}")
    for i in range(NUM_OBSTRUCTIONS):
        ob = extract_obstruction(obs, i)
        print(f"  obs{i}: ({ob['x1']:.3f},{ob['y1']:.3f},{ob['x2']:.3f},{ob['y2']:.3f}) static={ob['static']:.0f}")

    action = approach.get_action(obs)
    print(f"  action={action}")
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"DONE at step={step+1}")
        break

env.close()
