import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, block_center_from_obs, shelf_inner_bounds, BLOCK_NAMES, is_block_in_shelf

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')

for seed in [5, 12, 100]:
    obs, info = env.reset(seed=seed)
    robot = extract_robot(obs)
    approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
    approach.reset(obs, info)
    print(f"=== Seed {seed} ===")
    print(f"Init robot: ({robot.x:.3f},{robot.y:.3f}) theta={robot.theta:.3f} arm={robot.arm_joint:.3f}")
    sx_min,sx_max,sy_min,sy_max = shelf_inner_bounds(obs)
    print(f"Shelf: x=[{sx_min:.3f},{sx_max:.3f}]")
    for bn in BLOCK_NAMES:
        cx,cy = block_center_from_obs(obs,bn)
        print(f"  {bn}: ({cx:.3f},{cy:.3f}) in_shelf={is_block_in_shelf(obs,bn)}")
    
    cur = approach._current
    print(f"First behavior: {type(cur).__name__}, align_target={getattr(cur,'_align_target',None):.3f}")
    print(f"  target_robot: ({getattr(cur,'_target_robot_x',None):.3f},{getattr(cur,'_target_robot_y',None):.3f})")
    print(f"  arm_target: {getattr(cur,'_arm_target',None):.3f}")
    
    # Run first 50 steps and track theta
    thetas = [robot.theta]
    prev = robot.theta
    for i in range(50):
        action = approach.get_action(obs)
        obs2, _, t, tr, _ = env.step(action)
        r2 = extract_robot(obs2)
        if abs(r2.theta - prev) > 0.001:
            print(f"  step {i}: theta {prev:.4f} -> {r2.theta:.4f} (action dtheta={action[2]:.4f})")
        prev = r2.theta
        obs = obs2
    print(f"  After 50 steps: theta={prev:.4f}")
    print()
