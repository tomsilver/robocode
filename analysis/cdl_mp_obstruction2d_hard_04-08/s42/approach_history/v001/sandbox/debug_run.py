import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import *
from act_helpers import *

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}

seed = 42  # no obstructions, goes straight to pickup
obs, info = env.reset(seed=seed)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

block = extract_target_block(obs)
surf = extract_target_surface(obs)
robot = extract_robot(obs)
print(f"Block: x={block['x']:.3f}, y={block['y']:.3f}, w={block['width']:.3f}, h={block['height']:.3f}")
print(f"  If center: top={block['y']+block['height']/2:.3f}, bottom={block['y']-block['height']/2:.3f}")
print(f"Surf: x={surf['x']:.3f}, y={surf['y']:.3f}, w={surf['width']:.3f}, h={surf['height']:.3f}")
print(f"Robot: x={robot['x']:.3f}, y={robot['y']:.3f}, arm_length={robot['arm_length']:.3f}")
print(f"approach_xy_for_pick: {approach_xy_for_pick(block, robot['arm_length'])}")
print(f"any_obs_on_surf: {any_obstruction_on_surface(obs)}")
print(f"is_holding: {is_holding(obs)}")
print()

max_steps = 500
prev_phase = None
for step in range(max_steps):
    b = approach._current
    phase = b._phase if hasattr(b, '_phase') else '?'
    robot = extract_robot(obs)
    
    if phase != prev_phase or step % 50 == 0:
        block = extract_target_block(obs)
        print(f"Step {step:4d}: phase={phase}, robot=({robot['x']:.3f},{robot['y']:.3f}), arm={robot['arm_joint']:.3f}, vac={robot['vacuum']:.0f}, block=({block['x']:.3f},{block['y']:.3f})")
        prev_phase = phase
    
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"DONE at step {step}! terminated={terminated}")
        break

if step == max_steps - 1:
    print(f"Failed after {max_steps} steps")
    robot = extract_robot(obs)
    block = extract_target_block(obs)
    print(f"Final: phase={approach._current._phase}, robot=({robot['x']:.3f},{robot['y']:.3f}), block=({block['x']:.3f},{block['y']:.3f})")
