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
env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

# Fast forward to step 11 (stuck position)
for step in range(11):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)

robot = get_robot(obs)
print(f"At stuck pos: ({robot['x']:.4f},{robot['y']:.4f}), arm={robot['arm_joint']:.3f}, theta={robot['theta']:.3f}")

# Try pure -y only (no x)
print("--- Testing pure DOWN from stuck position ---")
for step in range(10):
    robot = get_robot(obs)
    action = clip_action(dx=0.0, dy=-0.05, dtheta=0.0, darm=0.0, vac=0.0)
    obs, reward, terminated, truncated, info = env.step(action)
    robot2 = get_robot(obs)
    print(f"  step {step}: ({robot['x']:.4f},{robot['y']:.4f}) → ({robot2['x']:.4f},{robot2['y']:.4f})")

print("\n--- Resetting and trying pure LEFT from stuck position ---")
env2 = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs2, info2 = env2.reset(seed=0)
approach2 = GeneratedApproach(env2.action_space, env2.observation_space, primitives)
approach2.reset(obs2, info2)
for step in range(11):
    action2 = approach2.get_action(obs2)
    obs2, reward2, terminated2, truncated2, info2 = env2.step(action2)

for step in range(10):
    robot = get_robot(obs2)
    action2 = clip_action(dx=-0.05, dy=0.0, dtheta=0.0, darm=0.0, vac=0.0)
    obs2, reward2, terminated2, truncated2, info2 = env2.step(action2)
    robot2 = get_robot(obs2)
    print(f"  step {step}: ({robot['x']:.4f},{robot['y']:.4f}) → ({robot2['x']:.4f},{robot2['y']:.4f})")
