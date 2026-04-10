import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_target_block, approach_xy_for_pick

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}

obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

last_phase = None
for steps in range(700):
    cur = approach._current
    phase = getattr(cur, '_phase', 'N/A')
    
    if type(cur).__name__ == 'PickupTargetBlock' and phase != last_phase:
        r = extract_robot(obs)
        blk = extract_target_block(obs)
        pick_x, pick_y = approach_xy_for_pick(blk, r['arm_length'])
        print(f"step={steps} phase={phase}: blk=({blk['x']:.4f},{blk['y']:.4f},{blk['width']:.4f},{blk['height']:.4f}) arm_len={r['arm_length']:.4f}")
        print(f"  pick_x={pick_x:.4f} pick_y={pick_y:.4f}  self._pick_x={getattr(cur,'_pick_x',None):.4f} self._pick_y={getattr(cur,'_pick_y',None):.4f}")
        print(f"  robot_y={r['y']:.4f} robot_aj={r['arm_joint']:.4f} vac={r['vacuum']:.0f}")
        last_phase = phase

    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print("Done")
