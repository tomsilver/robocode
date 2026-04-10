import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction

primitives = {"BiRRT": BiRRT}
env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

for step in range(57):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)

r = extract_robot(obs)
print(f"At extend start: robot=({r['x']:.4f},{r['y']:.4f}) theta={r['theta']:.4f} arm={r['arm_joint']:.4f}")
ob = extract_obstruction(obs, 0)
print(f"Obs0: x1={ob['x1']:.4f} x2={ob['x2']:.4f} y1={ob['y1']:.4f} y2={ob['y2']:.4f}")
GRIPPER_HALF_ALONG = 0.035
# Calculate: when arm_joint=0.16, gripper front at robot_x + 0.16 + 0.035
print(f"  Arm contact at 0.16: gripper_front = {r['x']+0.16+GRIPPER_HALF_ALONG:.4f} vs obs0 x1={ob['x1']:.4f}")
print(f"  arm_joint for contact = {ob['x1'] - r['x'] - GRIPPER_HALF_ALONG:.4f}")

for step in range(57, 150):
    r = extract_robot(obs)
    if step % 5 == 0:
        print(f"  step={step} arm={r['arm_joint']:.4f} vac={r['vacuum']:.0f} theta={r['theta']:.4f}")
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
