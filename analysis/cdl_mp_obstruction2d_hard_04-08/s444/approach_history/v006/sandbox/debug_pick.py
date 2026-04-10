"""Debug obstruction pick with new constants."""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'primitives')

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_obstruction, get_robot, obstruction_overlaps_surface
from behaviors import PickOneObstruction
import numpy as np

from motion_planning import BiRRT
primitives = {'BiRRT': BiRRT}

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

b = PickOneObstruction(primitives, 0)
b.reset(obs)
print(f"pick_y={b._pick_y:.4f} obs0_top=0.161 drop_x={b._drop_x:.3f}")
print(f"OBS_PICK_ARM=0.175")

for step in range(200):
    action = b.step(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    r = get_robot(obs)
    o0 = get_obstruction(obs, 0)

    if step % 20 == 0 or b._phase in ('grasp', 'retract', 'nav_high2') or step < 5:
        print(f"step={step:3d} phase={b._phase} "
              f"robot=({r['x']:.3f},{r['y']:.3f}) arm={r['arm_joint']:.4f} theta={r['theta']:.3f} "
              f"obs0=({o0['x']:.3f},{o0['y']:.3f}) on={obstruction_overlaps_surface(obs,0)}")
    if b.terminated(obs):
        print(f"OBS0 CLEARED at step {step}!")
        break
    if terminated or truncated:
        break
