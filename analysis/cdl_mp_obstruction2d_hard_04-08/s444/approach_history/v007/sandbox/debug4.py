import numpy as np
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import get_robot, get_obstruction
from act_helpers import clip_action

primitives = {'BiRRT': BiRRT}

def setup_stuck():
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=0)
    approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
    approach.reset(obs, info)
    for step in range(11):
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
    return env, obs

# Test UP
env, obs = setup_stuck()
robot = get_robot(obs)
print(f"Stuck at ({robot['x']:.4f},{robot['y']:.4f})")
print("Testing UP:")
for step in range(5):
    robot = get_robot(obs)
    action = clip_action(dx=0.0, dy=0.05, dtheta=0.0, darm=0.0, vac=0.0)
    obs, _, _, _, _ = env.step(action)
    robot2 = get_robot(obs)
    print(f"  ({robot['x']:.4f},{robot['y']:.4f}) → ({robot2['x']:.4f},{robot2['y']:.4f})")

# Test: retract arm first, then go down
env, obs = setup_stuck()
print("\nTesting: RETRACT ARM then go down:")
for step in range(5):
    robot = get_robot(obs)
    action = clip_action(dx=0.0, dy=0.0, dtheta=0.0, darm=-0.05, vac=0.0)
    obs, _, _, _, _ = env.step(action)
    robot2 = get_robot(obs)
    print(f"  arm_change: ({robot['arm_joint']:.3f}) → ({robot2['arm_joint']:.3f}), pos: ({robot2['x']:.4f},{robot2['y']:.4f})")

# Test small dx=+0.01 and dy=-0.05 (move right while descending)
env, obs = setup_stuck()
print("\nTesting: RIGHT+DOWN (dx=0.05, dy=-0.05):")
for step in range(10):
    robot = get_robot(obs)
    action = clip_action(dx=0.05, dy=-0.05, dtheta=0.0, darm=0.0, vac=0.0)
    obs, _, _, _, _ = env.step(action)
    robot2 = get_robot(obs)
    print(f"  ({robot['x']:.4f},{robot['y']:.4f}) → ({robot2['x']:.4f},{robot2['y']:.4f})")
