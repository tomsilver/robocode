import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_target_block, extract_target_surface, extract_obstruction, NUM_OBSTRUCTIONS

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}

for seed in [1, 2]:
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
    approach.reset(obs, info)
    done = False
    steps = 0
    last_phase = None
    phase_count = 0
    while not done and steps < 2000:
        cur = approach._current
        phase = getattr(cur, '_phase', 'N/A')
        if phase != last_phase:
            r = extract_robot(obs)
            print(f"  [s={seed} step={steps}] {type(cur).__name__} phase={phase} r=({r['x']:.3f},{r['y']:.3f},{r['arm_joint']:.3f}) vac={r['vacuum']:.0f}")
            last_phase = phase; phase_count = 0
        phase_count += 1
        if phase_count == 200:
            r = extract_robot(obs)
            print(f"  STUCK step={steps} phase={phase} robot=({r['x']:.3f},{r['y']:.3f}) vac={r['vacuum']:.0f}")
            blk = extract_target_block(obs)
            print(f"  block=({blk['x']:.3f},{blk['y']:.3f},{blk['width']:.3f},{blk['height']:.3f})")
            for i in range(NUM_OBSTRUCTIONS):
                ob = extract_obstruction(obs, i)
                print(f"  obs[{i}]=({ob['x']:.3f},{ob['y']:.3f},{ob['width']:.3f},{ob['height']:.3f}) cy={ob['y']+ob['height']/2:.3f}")
            break
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    if done:
        print(f"  SOLVED seed={seed} in {steps} steps")
    print()
