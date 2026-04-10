import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, extract_target_block, extract_target_surface, NUM_OBSTRUCTIONS

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}

for seed in [0, 1]:
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
    approach.reset(obs, info)

    done = False
    steps = 0
    max_steps = 2000
    last_phase = None
    phase_count = 0

    while not done and steps < max_steps:
        cur = approach._current
        phase = getattr(cur, '_phase', 'N/A')
        if phase != last_phase:
            r = extract_robot(obs)
            print(f"  [seed={seed} step={steps}] behavior={type(cur).__name__} phase={phase} robot=({r['x']:.3f},{r['y']:.3f}) theta={r['theta']:.3f} aj={r['arm_joint']:.3f} vac={r['vacuum']:.0f}")
            if hasattr(cur, '_target_idx'):
                ti = cur._target_idx
                if ti is not None and ti >= 0:
                    ob = extract_obstruction(obs, ti)
                    print(f"    target_idx={ti} ob=({ob['x']:.3f},{ob['y']:.3f},{ob['width']:.3f},{ob['height']:.3f})")
            last_phase = phase
            phase_count = 0
        phase_count += 1
        if phase_count == 300:
            r = extract_robot(obs)
            print(f"  [seed={seed} step={steps}] STUCK in phase={phase} robot=({r['x']:.3f},{r['y']:.3f}) theta={r['theta']:.3f} aj={r['arm_joint']:.3f}")
            if hasattr(cur, '_target_idx'):
                ti = cur._target_idx
                if ti is not None and ti >= 0:
                    ob = extract_obstruction(obs, ti)
                    print(f"    target_idx={ti} ob=({ob['x']:.3f},{ob['y']:.3f},{ob['width']:.3f},{ob['height']:.3f})")
                    print(f"    drop_x={getattr(cur, '_drop_x', 'N/A')}")
            break

        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    if not done:
        r = extract_robot(obs)
        print(f"FAILED seed={seed} final: behavior={type(approach._current).__name__} phase={getattr(approach._current,'_phase','N/A')} robot=({r['x']:.3f},{r['y']:.3f},{r['arm_joint']:.3f}) vac={r['vacuum']:.0f}")
        blk = extract_target_block(obs)
        surf = extract_target_surface(obs)
        print(f"  block=({blk['x']:.3f},{blk['y']:.3f}), surf=({surf['x']:.3f},{surf['y']:.3f},{surf['width']:.3f},{surf['height']:.3f})")
        for i in range(NUM_OBSTRUCTIONS):
            ob = extract_obstruction(obs, i)
            print(f"  obs[{i}]=({ob['x']:.3f},{ob['y']:.3f},{ob['width']:.3f},{ob['height']:.3f})")
