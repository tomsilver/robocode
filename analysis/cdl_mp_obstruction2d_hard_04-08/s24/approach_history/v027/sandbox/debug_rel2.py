import sys; sys.path.insert(0,"primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, NUM_OBSTRUCTIONS

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives={"BiRRT": BiRRT})
approach.reset(obs, info)

for step in range(145):
    approach.get_action(obs)
    obs, *_ = env.step(approach.get_action(obs))

for step in range(145, 200):
    cur = approach._current
    r = extract_robot(obs)
    action = approach.get_action(obs)
    obs_positions = [extract_obstruction(obs, i) for i in range(NUM_OBSTRUCTIONS)]
    print(f"step={step} i={getattr(cur,'_i',0)} {cur._phase} theta={r['theta']:.3f} rc={getattr(cur,'_release_count',0)} act_dth={action[2]:.3f}")
    for i, op in enumerate(obs_positions):
        print(f"  obs{i}:({op['x1']:.3f},{op['y1']:.3f},{op['x2']:.3f},{op['y2']:.3f})")
    obs, *_ = env.step(action)
    if cur._phase != approach._current._phase:
        print(f"  -> {approach._current._phase}"); break
env.close()
