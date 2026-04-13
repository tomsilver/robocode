import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import get_outside_blocks, is_block_in_shelf, BLOCK_NAMES, extract_robot, block_center_from_obs, shelf_inner_bounds

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')

for seed in [1, 3, 42]:
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
    approach.reset(obs, info)
    
    done = False
    steps = 0
    max_steps = 2000
    last_phase = None
    phase_start = 0
    
    while not done and steps < max_steps:
        cur = approach._current
        phase = getattr(cur, '_phase', '?') if cur else 'None'
        cname = type(cur).__name__ if cur else 'None'
        label = f"{cname}.{phase}"
        if label != last_phase:
            if last_phase is not None:
                print(f"  [{phase_start}-{steps}] {last_phase}")
            last_phase = label
            phase_start = steps
        
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    
    if last_phase:
        print(f"  [{phase_start}-{steps}] {last_phase}")
    
    robot = extract_robot(obs)
    in_shelf = [bn for bn in BLOCK_NAMES if is_block_in_shelf(obs, bn)]
    sx_min, sx_max, sy_min, sy_max = shelf_inner_bounds(obs)
    print(f"Seed {seed}: steps={steps}, terminated={terminated}, in_shelf={in_shelf}")
    print(f"  robot=({robot.x:.3f},{robot.y:.3f}) theta={robot.theta:.3f} arm={robot.arm_joint:.3f} vac={robot.vacuum:.1f}")
    for bn in BLOCK_NAMES:
        cx, cy = block_center_from_obs(obs, bn)
        in_s = is_block_in_shelf(obs, bn)
        print(f"  {bn}: center=({cx:.3f},{cy:.3f}) in_shelf={in_s}")
    print(f"  shelf inner: x=[{sx_min:.3f},{sx_max:.3f}] y=[{sy_min:.3f},{sy_max:.3f}]")
    print()
