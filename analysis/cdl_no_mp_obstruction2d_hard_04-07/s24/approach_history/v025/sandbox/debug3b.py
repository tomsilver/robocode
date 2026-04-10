import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, {})
approach.reset(obs, info)

for step in range(100):
    cur = approach._current
    phase = cur._phase
    beh_name = type(cur).__name__
    r = extract_robot(obs)
    obs3 = extract_obstruction(obs, 3)
    action = approach.get_action(obs)
    if step >= 44:
        print(f"step {step}: beh={beh_name[:4]}, phase={phase}, robot=({r['x']:.3f},{r['y']:.3f}), arm={r['arm_joint']:.3f}, vac={r['vacuum']:.1f}, obs3=({obs3['x']:.3f},{obs3['y']:.3f})")
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("Done!")
        break
