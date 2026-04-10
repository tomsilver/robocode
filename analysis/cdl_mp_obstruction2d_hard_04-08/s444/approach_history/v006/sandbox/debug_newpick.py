"""Test new PickOneObstruction strategy."""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction, obstruction_overlaps_surface
from behaviors import PickOneObstruction
import numpy as np

for seed in [0, 1, 2]:
    print(f"\n=== Seed {seed} ===")
    env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
    obs, _ = env.reset(seed=seed)

    # Find which obs are on surface
    from obs_helpers import any_obstruction_on_surface, NUM_OBSTRUCTIONS
    on_surf = [i for i in range(NUM_OBSTRUCTIONS) if obstruction_overlaps_surface(obs, i)]
    print(f"  Obs on surface: {on_surf}")

    for obs_idx in on_surf[:1]:  # test first obs
        o = get_obstruction(obs, obs_idx)
        obs_top = o['y'] + o['height']/2
        from behaviors import OBS_PICK_ARM, OBS_PICK_CLEARANCE
        pick_y = obs_top + OBS_PICK_ARM + OBS_PICK_CLEARANCE
        print(f"  obs{obs_idx}: top={obs_top:.4f} pick_y={pick_y:.4f}")
        print(f"  suction_cy at pick_y: {pick_y-OBS_PICK_ARM-0.015:.4f} (want < {obs_top:.4f})")

        b = PickOneObstruction(None, obs_idx)
        b.reset(obs)

        for step in range(300):
            action = b.step(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            if step % 30 == 0:
                r = get_robot(obs)
                o2 = get_obstruction(obs, obs_idx)
                arm_tip_y = r['y'] - r['arm_joint']
                on = obstruction_overlaps_surface(obs, obs_idx)
                print(f"    step={step:3d} ph={b._phase:12s} y={r['y']:.4f} arm={r['arm_joint']:.4f} "
                      f"arm_tip={arm_tip_y:.4f} obs_y={o2['y']:.4f} on={on}")

            if b.terminated(obs):
                print(f"    CLEARED at step {step}!")
                break
            if terminated or truncated:
                print(f"    TERMINATED/TRUNCATED")
                break
