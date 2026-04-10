import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, extract_target_block, NUM_OBSTRUCTIONS, is_obstruction_blocking_surface

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}

for seed in [0, 1]:
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
    approach.reset(obs, info)
    done = False; steps = 0

    while not done and steps < 350:
        cur = approach._current
        phase = getattr(cur, '_phase', '?')
        cls = cur.__class__.__name__
        r = extract_robot(obs)

        if phase == 'NAV_DOWN' and steps > 50 and steps % 15 == 0:
            print(f"seed={seed} step={steps} [{cls}] NAV_DOWN robot=({r['x']:.3f},{r['y']:.3f})")
            if cur._path_step < len(cur._path):
                wp = cur._path[cur._path_step]
                print(f"  wp=({wp[0]:.3f},{wp[1]:.3f}) pick_y={getattr(cur,'_pick_y',None)}")
            for i in range(NUM_OBSTRUCTIONS):
                ob = extract_obstruction(obs, i)
                print(f"  obs{i}: y=[{ob['y']:.3f},{ob['y']+ob['height']:.3f}] x=[{ob['x']:.3f},{ob['x']+ob['width']:.3f}]")
            blk = extract_target_block(obs)
            print(f"  block: y=[{blk['y']:.3f},{blk['y']+blk['height']:.3f}] x=[{blk['x']:.3f},{blk['x']+blk['width']:.3f}]")

        if phase == 'NAV_HZDROP' and steps > 50 and steps % 15 == 0:
            print(f"seed={seed} step={steps} [{cls}] NAV_HZDROP robot=({r['x']:.3f},{r['y']:.3f}) path_step={cur._path_step}/{len(cur._path)}")
            if cur._path_step < len(cur._path):
                wp = cur._path[cur._path_step]
                print(f"  wp=({wp[0]:.3f},{wp[1]:.3f}) drop_x={cur._drop_x:.3f}")

        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    print(f"Seed {seed}: steps={steps} done={done}")
