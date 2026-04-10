"""Debug approach on seed 42."""
import sys
sys.path.insert(0, '/sandbox')
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

print("Initial state:")
robot = extract_robot(obs)
print(f"  Robot: x={robot['x']:.3f}, y={robot['y']:.3f}, theta={robot['theta']:.3f}")
print(f"  Any obstruction on surface: {any_obstruction_on_surface(obs)}")
for i in range(NUM_OBSTRUCTIONS):
    ob = extract_obstruction(obs, i)
    on = is_obstruction_on_surface(obs, i)
    print(f"  Obs{i}: x={ob['x']:.3f}, y={ob['y']:.3f}, w={ob['width']:.3f}, h={ob['height']:.3f}, on_surf={on}")
surf = extract_target_surface(obs)
print(f"  Surface: x={surf['x']:.3f}, y={surf['y']:.3f}, w={surf['width']:.3f}, h={surf['height']:.3f}")

print(f"\nCurrent behavior: {approach._current.__class__.__name__}")

# Run a few steps
for step in range(100):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    robot = extract_robot(obs)
    tip = gripper_tip_xy(obs)
    phase = approach._current._phase if hasattr(approach._current, '_phase') else '?'
    if step % 20 == 0:
        print(f"Step {step:3d}: robot=({robot['x']:.3f},{robot['y']:.3f}), theta={robot['theta']:.3f}, arm={robot['arm_joint']:.3f}, vac={robot['vacuum']:.0f}, phase={phase}")
    if terminated:
        print(f"SUCCESS at step {step}")
        break
