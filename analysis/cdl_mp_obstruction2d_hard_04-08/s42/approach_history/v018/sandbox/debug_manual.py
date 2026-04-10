import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, extract_target_block, NUM_OBSTRUCTIONS
from act_helpers import make_action

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}
obs, info = env.reset(seed=1)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

# Run until NAV_PLACE starts and robot reaches y=0.45
count = 0
for step in range(260):
    cur = approach._current
    bname = type(cur).__name__
    ph = getattr(cur, '_phase', '?')
    r = extract_robot(obs)
    if bname == 'PlaceTargetBlock' and ph == 'NAV_PLACE':
        count += 1
        if count == 9:  # just stuck
            print(f"At step {step}, robot at ({r['x']:.4f},{r['y']:.4f}), count={count}")
            # Try manually moving down
            for manual in range(5):
                act = make_action(dy=-0.05, vac=1.0)
                obs2, _, _, _, _ = env.step(act)
                r2 = extract_robot(obs2)
                print(f"  Manual step {manual+1}: robot=({r2['x']:.4f},{r2['y']:.4f})")
                # Check obstructions
                for i in range(NUM_OBSTRUCTIONS):
                    o = extract_obstruction(obs2, i)
                    print(f"    obs[{i}]: bl=({o['x']:.4f},{o['y']:.4f})")
                blk = extract_target_block(obs2)
                print(f"    block: bl=({blk['x']:.4f},{blk['y']:.4f}) w={blk['width']:.4f} h={blk['height']:.4f}")
                obs2_copy = obs2
            break
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
