import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, block_center_from_obs, extract_rect, BLOCK_NAMES

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')

obs, info = env.reset(seed=100)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

r = extract_robot(obs)
print(f"Init: robot=({r.x:.4f},{r.y:.4f}) theta={r.theta:.4f}")
for bn in BLOCK_NAMES:
    rect = extract_rect(obs, bn)
    cx, cy = block_center_from_obs(obs, bn)
    print(f"  {bn}: corner=({rect.x:.4f},{rect.y:.4f}) theta={rect.theta:.4f} center=({cx:.4f},{cy:.4f})")

for step in range(15):
    action = approach.get_action(obs)
    obs_new, rew, t, tr, _ = env.step(action)
    r_new = extract_robot(obs_new)
    rejected = (abs(r_new.x - r.x) < 1e-6 and abs(r_new.y - r.y) < 1e-6 
                and abs(r_new.theta - r.theta) < 1e-6 and abs(r_new.arm_joint - r.arm_joint) < 1e-6)
    print(f"Step {step}: action=[{action[0]:.4f},{action[1]:.4f},{action[2]:.4f},{action[3]:.4f},{action[4]:.1f}] -> theta={r_new.theta:.4f} {'REJECTED' if rejected else 'OK'}")
    # Print block positions if changed
    for bn in BLOCK_NAMES:
        r_old = extract_rect(obs, bn)
        r_nw = extract_rect(obs_new, bn)
        if abs(r_old.x - r_nw.x) > 0.001 or abs(r_old.y - r_nw.y) > 0.001:
            cx_new, cy_new = block_center_from_obs(obs_new, bn)
            print(f"  {bn} MOVED to center ({cx_new:.4f},{cy_new:.4f})")
    obs = obs_new
    r = r_new
