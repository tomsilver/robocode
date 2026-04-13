import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import get_outside_blocks, is_block_in_shelf, BLOCK_NAMES, extract_robot

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')

obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

print("Behavior sequence:")
import collections
q = approach._behaviors.copy()
cur = approach._current
print(f"  0: {type(cur).__name__}")
i = 1
for b in list(q):
    print(f"  {i}: {type(b).__name__}")
    i += 1

done = False
steps = 0
max_steps = 2000
prev_behavior = type(approach._current).__name__
prev_phase = approach._current._phase

for step in range(max_steps):
    cur_bname = type(approach._current).__name__
    cur_phase = approach._current._phase
    if cur_bname != prev_behavior or cur_phase != prev_phase:
        robot = extract_robot(obs)
        in_shelf = [b for b in BLOCK_NAMES if is_block_in_shelf(obs, b)]
        print(f"  step={step}: {prev_behavior}.{prev_phase} -> {cur_bname}.{cur_phase} | robot=({robot.x:.2f},{robot.y:.2f}) arm={robot.arm_joint:.3f} vac={robot.vacuum:.1f} in_shelf={in_shelf}")
        prev_behavior = cur_bname
        prev_phase = cur_phase
    
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        print(f"  DONE at step={step+1}, terminated={terminated}")
        break

robot = extract_robot(obs)
in_shelf = [b for b in BLOCK_NAMES if is_block_in_shelf(obs, b)]
print(f"Final: robot=({robot.x:.2f},{robot.y:.2f}) arm={robot.arm_joint:.3f} vac={robot.vacuum:.1f} in_shelf={in_shelf}")
