import numpy as np
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import get_robot, get_obstruction, any_obstruction_on_surface, is_holding_block, block_on_surface

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

for step in range(100):
    robot = get_robot(obs)
    cur = approach._current
    phase = getattr(cur, '_phase', '?')
    sub = getattr(cur, '_current', None)
    sub_phase = getattr(sub, '_phase', '?') if sub else '?'
    if step % 5 == 0:
        print(f"step {step:3d}: robot=({robot['x']:.3f},{robot['y']:.3f}) arm={robot['arm_joint']:.3f} vac={robot['vacuum']:.1f} phase={type(cur).__name__}:{phase} sub={sub_phase}")
        for i in range(4):
            o = get_obstruction(obs, i)
            print(f"  obs{i}: x={o['x']:.3f} y={o['y']:.3f} w={o['width']:.3f} h={o['height']:.3f}")
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Done at step {step+1}!")
        break
