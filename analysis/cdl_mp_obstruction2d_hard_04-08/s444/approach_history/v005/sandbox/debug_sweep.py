"""Debug sweep - step-level detail."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_obstruction, get_surface, get_robot, obstruction_overlaps_surface, NUM_OBSTRUCTIONS
from act_helpers import NAV_HIGH_Y
from behaviors import ClearAllObstructions
from act_helpers import clip_action
import numpy as np

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

from motion_planning import BiRRT
primitives = {'BiRRT': BiRRT}

b = ClearAllObstructions(primitives)
b.reset(obs)

print("Initial state:")
r = get_robot(obs)
print(f"  Robot: ({r['x']:.3f}, {r['y']:.3f}) theta={r['theta']:.3f}")
for i in range(4):
    o = get_obstruction(obs, i)
    print(f"  Obs{i}: ({o['x']:.3f}, {o['y']:.3f}) w={o['width']:.3f} h={o['height']:.3f}")

print("\n--- Steps ---")
for step in range(120):
    action = b.step(obs)
    prev_r = get_robot(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    r = get_robot(obs)

    if step < 30 or (step % 20 == 0):
        o0 = get_obstruction(obs, 0)
        print(f"step={step:3d} phase={b._phase} robot=({r['x']:.3f},{r['y']:.3f}) "
              f"theta={r['theta']:.2f} action=({action[0]:.3f},{action[1]:.3f},{action[2]:.3f}) "
              f"obs0=({o0['x']:.3f},{o0['y']:.3f}) obs_on={obstruction_overlaps_surface(obs,0)}")
    if terminated or truncated:
        print(f"Done at step {step}")
        break
