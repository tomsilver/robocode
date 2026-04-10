import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, {})
approach.reset(obs, info)

for step in range(90):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)

# Now debug step by step
for step in range(90, 100):
    cur = approach._current
    r = extract_robot(obs)
    o0 = extract_obstruction(obs, 0)
    action = approach.get_action(obs)
    print(f"step {step}: phase={cur._phase}, robot=({r['x']:.3f},{r['y']:.3f}), arm={r['arm_joint']:.3f}, obs0=({o0['x']:.3f},{o0['y']:.3f}), action={action[:4]}, pick_y={cur._pick_y:.3f}")
    obs, reward, terminated, truncated, info = env.step(action)
