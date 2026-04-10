import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, extract_target_surface, extract_target_block, obstruction_on_surface, NUM_OBSTRUCTIONS, get_drop_zones

seed = 20
env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=seed)
approach = GeneratedApproach(env.action_space, env.observation_space, {})
approach.reset(obs, info)

surf = extract_target_surface(obs)
blk = extract_target_block(obs)
print(f"Surface: x={surf['x']:.3f}-{surf['x']+surf['width']:.3f} y={surf['y']:.3f} h={surf['height']:.3f}")
print(f"Block: x={blk['x']:.3f} y={blk['y']:.3f} w={blk['width']:.3f} h={blk['height']:.3f}")
for i in range(NUM_OBSTRUCTIONS):
    o = extract_obstruction(obs, i)
    on_surf = obstruction_on_surface(obs, i)
    print(f"Obs{i}: x={o['x']:.3f}-{o['x']+o['width']:.3f} y={o['y']:.3f} h={o['height']:.3f} on_surf={on_surf}")
zones = get_drop_zones(obs)
print(f"Drop zones: {[f'{z:.3f}' for z in zones]}")

done = False
steps = 0
last_phase = None; last_cls = None
while not done and steps < 2000:
    cur = approach._current
    cls = type(cur).__name__
    if hasattr(cur, '_phase'):
        ph = cur._phase
        if cls != last_cls or ph != last_phase:
            r = extract_robot(obs)
            print(f"  step={steps} {cls} phase={ph} rx={r['x']:.3f} ry={r['y']:.3f}")
            last_phase = ph; last_cls = cls
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1
    done = terminated or truncated
r = extract_robot(obs)
print(f"Final: steps={steps} success={terminated} rx={r['x']:.3f} ry={r['y']:.3f}")
