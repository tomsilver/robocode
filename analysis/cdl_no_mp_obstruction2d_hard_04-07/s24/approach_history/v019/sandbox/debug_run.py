import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_target_block, extract_target_surface, obstruction_on_surface, block_is_on_surface

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, {})
approach.reset(obs, info)

for step in range(400):
    r = extract_robot(obs)
    blk = extract_target_block(obs)
    surf = extract_target_surface(obs)
    obs_on = [obstruction_on_surface(obs, i) for i in range(4)]
    cur = approach._current
    phase = cur._phase
    beh_name = type(cur).__name__
    if step % 20 == 0 or step > 180:
        print(f"step {step}: beh={beh_name}, phase={phase}, robot=({r['x']:.3f},{r['y']:.3f}), arm={r['arm_joint']:.3f}, vac={r['vacuum']:.1f}, obs_on={obs_on}, block_on_surf={block_is_on_surface(obs)}")
        print(f"  blk=({blk['x']:.3f},{blk['y']:.3f}), surf=({surf['x']:.3f},{surf['y']:.3f},{surf['width']:.3f},{surf['height']:.3f})")
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Done at step {step+1}, terminated={terminated}")
        break
