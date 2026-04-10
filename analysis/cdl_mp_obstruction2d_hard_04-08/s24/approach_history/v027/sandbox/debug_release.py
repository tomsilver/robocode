import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot

primitives = {"BiRRT": BiRRT}
env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

for step in range(70):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)

print(f"Phase: {approach._current._phase}")
for step in range(70, 120):
    cur = approach._current
    r = extract_robot(obs)
    action = approach.get_action(obs)
    print(f"step={step} phase={cur._phase} theta={r['theta']:.4f} vac={r['vacuum']:.0f} rc={cur._release_count} action_dtheta={action[2]:.4f}")
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"DONE at step={step+1}"); break
    if type(cur).__name__ != type(approach._current).__name__ or cur._phase != approach._current._phase:
        print(f"  Phase changed to {approach._current._phase}"); break
env.close()
