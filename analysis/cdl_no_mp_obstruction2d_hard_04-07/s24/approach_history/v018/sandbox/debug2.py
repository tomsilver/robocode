import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_target_block, extract_target_surface, extract_obstruction, obstruction_on_surface
from act_helpers import make_action

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, {})
approach.reset(obs, info)

for step in range(200):
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if step == 98:
        # Print detailed state at the stuck point
        r = extract_robot(obs)
        print(f"Robot: x={r['x']:.4f}, y={r['y']:.4f}, theta={r['theta']:.3f}, arm={r['arm_joint']:.4f}, vac={r['vacuum']:.2f}")
        print(f"Action: {action}")
        for i in range(4):
            o = extract_obstruction(obs, i)
            print(f"obs{i}: x={o['x']:.4f}, y={o['y']:.4f}, w={o['width']:.4f}, h={o['height']:.4f}")
        blk = extract_target_block(obs)
        print(f"block: x={blk['x']:.4f}, y={blk['y']:.4f}, w={blk['width']:.4f}, h={blk['height']:.4f}")
        surf = extract_target_surface(obs)
        print(f"surf: x={surf['x']:.4f}, y={surf['y']:.4f}, w={surf['width']:.4f}, h={surf['height']:.4f}")
    if terminated or truncated:
        print(f"Done at step {step+1}")
        break
