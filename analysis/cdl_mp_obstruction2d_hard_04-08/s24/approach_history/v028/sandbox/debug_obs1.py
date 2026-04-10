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

# Run to ClearObstruction[1].release
for step in range(200):
    cur = approach._current
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"DONE at step={step+1}"); break

# Now trace release
for step in range(200, 280):
    cur = approach._current
    r = extract_robot(obs)
    action = approach.get_action(obs)
    print(f"step={step} {type(cur).__name__}[{getattr(cur,'_i',0)}].{cur._phase} theta={r['theta']:.4f} rc={getattr(cur,'_release_count',0)} drop=({getattr(cur,'_drop_x',0):.3f},{getattr(cur,'_drop_y',0):.3f}) robot=({r['x']:.3f},{r['y']:.3f})")
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"DONE at step={step+1}"); break
env.close()
