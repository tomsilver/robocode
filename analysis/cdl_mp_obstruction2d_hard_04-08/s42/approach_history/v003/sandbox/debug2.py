import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import *
from act_helpers import *

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}
obs, info = env.reset(seed=42)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

print("Behavior:", approach._current.__class__.__name__)
prev_phase = None
for step in range(500):
    robot = extract_robot(obs)
    phase = approach._current._phase if hasattr(approach._current, '_phase') else '?'
    bname = approach._current.__class__.__name__
    tip = gripper_tip_xy(obs)
    block = extract_target_block(obs)
    
    if phase != prev_phase:
        print(f"Step {step:3d}: [{bname}] → {phase} | robot=({robot['x']:.3f},{robot['y']:.3f}) arm={robot['arm_joint']:.2f} vac={robot['vacuum']:.0f}")
        print(f"         block=({block['x']:.3f},{block['y']:.3f}) tip=({tip[0]:.3f},{tip[1]:.3f})")
        prev_phase = phase
    
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        print(f"SUCCESS at step {step}")
        break
