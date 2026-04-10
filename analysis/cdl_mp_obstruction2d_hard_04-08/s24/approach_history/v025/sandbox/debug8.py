import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import extract_robot, gripper_tip_pos, IDX_ROBOT_GRIP_HEIGHT, IDX_ROBOT_GRIP_WIDTH

env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")
obs, info = env.reset(seed=0)
r = extract_robot(obs)
print(f"grip_height (obs[7]) = {obs[IDX_ROBOT_GRIP_HEIGHT]:.4f}")
print(f"grip_width  (obs[8]) = {obs[IDX_ROBOT_GRIP_WIDTH]:.4f}")
print(f"Robot: x={r['x']:.3f} y={r['y']:.3f} theta={r['theta']:.3f} arm={r['arm_joint']:.3f}")
tip = gripper_tip_pos(obs)
print(f"Gripper tip at arm_min: {tip}")
# manually check
gw = r["grip_width"]
gh = r["grip_height"]
print(f"grip_height={gh:.4f} grip_width={gw:.4f}")
reach = r["arm_joint"] + gw + gw/2
print(f"suction reach formula = arm_joint + 1.5*grip_width = {reach:.4f}")
print(f"suction reach = arm_joint + 1.5*grip_height = {r['arm_joint'] + 1.5*gh:.4f}")
