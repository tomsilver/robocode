import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_target_block, extract_target_surface, is_block_on_surface

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}
obs, info = env.reset(seed=1)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

for steps in range(2000):
    cur = approach._current
    phase = getattr(cur, '_phase', 'N/A')
    
    if type(cur).__name__ == 'PlaceTargetBlock' and phase in ['PLAN_PLACE','NAV_PLACE']:
        r = extract_robot(obs)
        blk = extract_target_block(obs)
        print(f"step={steps} phase={phase} r=({r['x']:.4f},{r['y']:.4f}) surf_x={getattr(cur,'_surf_x',0):.4f} place_y={getattr(cur,'_place_y',0):.4f} on_surf={is_block_on_surface(obs)}")
        print(f"  blk=({blk['x']:.4f},{blk['y']:.4f}) path_len={len(getattr(cur,'_path',[]))}")
        if steps > 250:
            break

    action = approach.get_action(obs)
    obs, rew, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"DONE at step {steps}")
        break
