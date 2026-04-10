"""Print full initial state for several seeds."""
import sys
sys.path.insert(0, '.')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_surface, get_block, get_obstruction, NUM_OBSTRUCTIONS, obstruction_overlaps_surface

for seed in [0, 1, 2]:
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, info = env.reset(seed=seed)
    s = get_surface(obs)
    b = get_block(obs)
    r = get_robot(obs)
    print(f"\n=== Seed {seed} ===")
    print(f"  Surface: x={s['x']:.3f} y={s['y']:.3f} w={s['width']:.3f} h={s['height']:.3f} top={s['y']+s['height']/2:.3f}")
    print(f"  Block:   x={b['x']:.3f} y={b['y']:.3f} w={b['width']:.3f} h={b['height']:.3f} top={b['y']+b['height']/2:.3f}")
    print(f"  Robot:   x={r['x']:.3f} y={r['y']:.3f}")
    for i in range(NUM_OBSTRUCTIONS):
        o = get_obstruction(obs, i)
        on = obstruction_overlaps_surface(obs, i)
        print(f"  Obs{i}: x={o['x']:.3f} y={o['y']:.3f} w={o['width']:.3f} h={o['height']:.3f} top={o['y']+o['height']/2:.3f} on_surf={on}")
