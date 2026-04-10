import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_block, extract_surface, extract_obstruction, obstruction_overlaps_surface

primitives = {"BiRRT": BiRRT}
env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

print("Initial state:")
r = extract_robot(obs)
blk = extract_block(obs)
surf = extract_surface(obs)
print(f"  Robot: x={r['x']:.3f} y={r['y']:.3f} theta={r['theta']:.3f} arm={r['arm_joint']:.3f}")
print(f"  Block: cx={blk['cx']:.3f} cy={blk['cy']:.3f} x1={blk['x1']:.3f} x2={blk['x2']:.3f}")
print(f"  Surf:  cx={surf['cx']:.3f} cy={surf['cy']:.3f} y2={surf['y2']:.3f}")
for i in range(4):
    ob = extract_obstruction(obs, i)
    overlap = obstruction_overlaps_surface(obs, i)
    print(f"  Obs{i}: cx={ob['cx']:.3f} cy={ob['cy']:.3f} overlaps={overlap}")

print(f"\nBehavior queue: {[type(b).__name__ for b in approach._behaviors]}")
print(f"Current: {type(approach._current).__name__} phase={approach._current._phase}")

last_phase = None
last_bname = None
for step in range(500):
    bname = type(approach._current).__name__
    phase = approach._current._phase
    if bname != last_bname or phase != last_phase:
        r = extract_robot(obs)
        print(f"  step={step} {bname}.{phase} robot=({r['x']:.3f},{r['y']:.3f}) arm={r['arm_joint']:.3f} vac={r['vacuum']:.0f}")
        last_phase = phase
        last_bname = bname
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print(f"TERMINATED at step {step}")
        break
    if truncated:
        print(f"TRUNCATED at step {step}")
        break
else:
    print("DID NOT TERMINATE after 500 steps")
    r = extract_robot(obs)
    blk = extract_block(obs)
    print(f"  Final: robot=({r['x']:.3f},{r['y']:.3f}) theta={r['theta']:.3f} arm={r['arm_joint']:.3f} vac={r['vacuum']:.0f}")
    print(f"  Block: ({blk['cx']:.3f},{blk['cy']:.3f})")
