import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach
from obs_helpers import extract_robot

seed = 2
env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=seed)
approach = GeneratedApproach(env.action_space, env.observation_space, {})
approach.reset(obs, info)

done = False
steps = 0
while not done and steps < 400:
    cur = approach._current
    if hasattr(cur, '_phase') and cur._phase == 11:  # REPOSITION
        r = extract_robot(obs)
        if steps % 5 == 0:
            print(f"step={steps} REPOSITION rx={r['x']:.3f} ry={r['y']:.3f} drop_x={cur._drop_x:.3f} shift_attempts={cur._shift_attempts}")
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1
    done = terminated or truncated

r = extract_robot(obs)
print(f"Final step={steps} rx={r['x']:.3f} ry={r['y']:.3f}")
