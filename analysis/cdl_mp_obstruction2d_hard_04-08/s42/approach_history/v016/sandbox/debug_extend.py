import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_target_block, extract_obstruction, NUM_OBSTRUCTIONS

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}

obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

done = False
steps = 0
second_extend_started = False

while not done and steps < 700:
    cur = approach._current
    phase = getattr(cur, '_phase', 'N/A')
    
    # Detect the second EXTEND (after the first grasp attempt fails)
    if type(cur).__name__ == 'PickupTargetBlock' and phase == 'EXTEND' and steps > 150:
        if not second_extend_started:
            second_extend_started = True
            print(f"=== SECOND EXTEND STARTED at step {steps} ===")
        r = extract_robot(obs)
        blk = extract_target_block(obs)
        print(f"  step={steps}: aj={r['arm_joint']:.4f} al={r['arm_length']:.4f} ry={r['y']:.4f} vac={r['vacuum']:.0f}")
        print(f"  block: x={blk['x']:.4f} y={blk['y']:.4f} w={blk['width']:.4f} h={blk['height']:.4f}")
        for i in range(NUM_OBSTRUCTIONS):
            ob = extract_obstruction(obs, i)
            print(f"  obs[{i}]: x={ob['x']:.4f} y={ob['y']:.4f} w={ob['width']:.4f} h={ob['height']:.4f} cx={ob['x']+ob['width']/2:.4f} cy={ob['y']+ob['height']/2:.4f}")
        action = approach.get_action(obs)
        print(f"  action: dx={action[0]:.4f} dy={action[1]:.4f} dtheta={action[2]:.4f} darm={action[3]:.4f} vac={action[4]:.1f}")
    else:
        action = approach.get_action(obs)
    
    if second_extend_started and steps > 175:
        break  # got enough info
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    steps += 1

