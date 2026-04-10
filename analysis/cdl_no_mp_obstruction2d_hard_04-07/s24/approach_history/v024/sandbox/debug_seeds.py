"""Debug seeds 1 and 2."""
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, extract_target_surface, extract_target_block, obstruction_on_surface, NUM_OBSTRUCTIONS

for seed in [1, 2]:
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, {})
    approach.reset(obs, info)

    # Print initial state
    surf = extract_target_surface(obs)
    blk = extract_target_block(obs)
    print(f"\n=== SEED {seed} ===")
    print(f"Surface: x={surf['x']:.3f} w={surf['width']:.3f} y={surf['y']:.3f} h={surf['height']:.3f}")
    print(f"Block: x={blk['x']:.3f} y={blk['y']:.3f} w={blk['width']:.3f} h={blk['height']:.3f}")
    for i in range(NUM_OBSTRUCTIONS):
        o = extract_obstruction(obs, i)
        on_surf = obstruction_on_surface(obs, i)
        print(f"Obs{i}: x={o['x']:.3f} w={o['width']:.3f} y={o['y']:.3f} h={o['height']:.3f} on_surf={on_surf}")

    from obs_helpers import get_drop_zones
    zones = get_drop_zones(obs)
    print(f"Drop zones: {[f'{z:.3f}' for z in zones]}")

    total_reward = 0
    done = False
    steps = 0
    max_steps = 2000
    last_phase = None
    last_behavior = None
    
    while not done and steps < max_steps:
        cur = approach._current
        cur_cls = type(cur).__name__
        if hasattr(cur, '_phase'):
            ph = cur._phase
            if cur_cls != last_behavior or ph != last_phase:
                r = extract_robot(obs)
                print(f"  step={steps} {cur_cls} phase={ph} rx={r['x']:.3f} ry={r['y']:.3f}")
                last_phase = ph
                last_behavior = cur_cls
        
        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    r = extract_robot(obs)
    print(f"seed={seed}: steps={steps} success={terminated} final rx={r['x']:.3f} ry={r['y']:.3f}")
