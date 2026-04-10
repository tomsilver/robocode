import sys; sys.path.insert(0, "primitives")
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

# Run until lower phase
for step in range(200):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if approach._current._phase == "lower":
        print(f"Entered lower at step {step+1}")
        break

# Trace lower
for step in range(30):
    r = extract_robot(obs)
    action = approach.get_action(obs)
    print(f"step={step} phase={approach._current._phase} y={r['y']:.4f} theta={r['theta']:.4f} action={action}")
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("DONE"); break
env.close()
