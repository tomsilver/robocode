import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_block, obstruction_overlaps_surface

primitives = {"BiRRT": BiRRT}
env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

last_phase = None
last_bname = None
for step in range(2000):
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
    print("DID NOT TERMINATE")
    r = extract_robot(obs)
    print(f"  Final: robot=({r['x']:.3f},{r['y']:.3f}) arm={r['arm_joint']:.3f} vac={r['vacuum']:.0f}")
    blk = extract_block(obs)
    print(f"  Block: ({blk['cx']:.3f},{blk['cy']:.3f})")
