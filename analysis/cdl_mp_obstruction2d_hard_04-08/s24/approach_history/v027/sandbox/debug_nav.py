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

for step in range(125): # get past nav_drop for obs[1]
    approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(approach.get_action(obs))

# trace nav_drop[1] for 20 steps
for step in range(125, 160):
    cur = approach._current
    r = extract_robot(obs)
    action = approach.get_action(obs)
    obs_list = [(extract_obstruction(obs, i)['x1'], extract_obstruction(obs, i)['y1'],
                 extract_obstruction(obs, i)['x2'], extract_obstruction(obs, i)['y2'])
                for i in range(NUM_OBSTRUCTIONS)]
    print(f"step={step} {type(cur).__name__}[{getattr(cur,'_i',0)}].{cur._phase} robot=({r['x']:.3f},{r['y']:.3f}) theta={r['theta']:.3f} drop=({getattr(cur,'_drop_x',0):.3f},{getattr(cur,'_drop_y',0):.3f})")
    for i, ob in enumerate(obs_list):
        print(f"  obs{i}=({ob[0]:.3f},{ob[1]:.3f},{ob[2]:.3f},{ob[3]:.3f})")
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("DONE"); break
    if cur._phase != approach._current._phase:
        print(f"  -> {approach._current._phase}"); break
env.close()
