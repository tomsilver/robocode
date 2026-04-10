"""Trace all 1000 steps for seed 0."""
import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_block, extract_surface

primitives = {"BiRRT": BiRRT}
env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

prev_phase = None
prev_class = None

for step in range(1000):
    cur = approach._current
    cur_class = type(cur).__name__
    cur_phase = cur._phase

    if cur_class != prev_class or cur_phase != prev_phase:
        r = extract_robot(obs)
        blk = extract_block(obs)
        surf = extract_surface(obs)
        print(f"step={step} {cur_class}.{cur_phase} robot=({r['x']:.3f},{r['y']:.3f}) theta={r['theta']:.3f} arm={r['arm_joint']:.3f} vac={r['vacuum']:.0f} blk=({blk['cx']:.3f},{blk['cy']:.3f})")
        if cur_class == 'PlaceBlock':
            print(f"  goal=({cur._goal_x:.3f},{cur._goal_y:.3f}) gtheta={cur._goal_theta:.3f}")
        prev_phase = cur_phase
        prev_class = cur_class

    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"DONE step={step+1} reward={reward} terminated={terminated}")
        break

r = extract_robot(obs)
blk = extract_block(obs)
surf = extract_surface(obs)
print(f"\nFinal: {type(approach._current).__name__}.{approach._current._phase}")
print(f"robot=({r['x']:.3f},{r['y']:.3f}) arm={r['arm_joint']:.3f} vac={r['vacuum']:.0f}")
print(f"blk=({blk['x1']:.3f},{blk['y1']:.3f},{blk['x2']:.3f},{blk['y2']:.3f}) surf=({surf['x1']:.3f},{surf['y1']:.3f},{surf['x2']:.3f},{surf['y2']:.3f})")
env.close()
