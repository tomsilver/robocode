"""Debug push strategy - see what's happening."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, 'primitives')

from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_obstruction, get_surface, get_robot, obstruction_overlaps_surface, NUM_OBSTRUCTIONS, ROB_RADIUS
from act_helpers import NAV_HIGH_Y
import numpy as np

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)

s = get_surface(obs)
r = get_robot(obs)
print(f"Surface: x={s['x']:.3f} y={s['y']:.3f} w={s['width']:.3f} h={s['height']:.3f}")
print(f"Robot: x={r['x']:.3f} y={r['y']:.3f} theta={r['theta']:.3f} arm={r['arm_joint']:.3f}")

for i in range(NUM_OBSTRUCTIONS):
    o = get_obstruction(obs, i)
    on = obstruction_overlaps_surface(obs, i)
    print(f"Obs{i}: x={o['x']:.3f} y={o['y']:.3f} w={o['width']:.3f} h={o['height']:.3f} top={o['y']+o['height']/2:.3f} on_surface={on}")

# Compute push params for obs0
o0 = get_obstruction(obs, 0)
surf_x = s['x']
obs_x = o0['x']
obs_w = o0['width']
PUSH_MARGIN = 0.12
PUSH_EXTRA = 0.55
PUSH_Y_val = 0.26

if obs_x <= surf_x:
    approach_x = obs_x + obs_w/2 + ROB_RADIUS + PUSH_MARGIN
    push_target_x = max(0.15, obs_x - PUSH_EXTRA)
    print(f"\nObs0 push LEFT: approach_x={approach_x:.3f}, push_target={push_target_x:.3f}")
else:
    approach_x = obs_x - obs_w/2 - ROB_RADIUS - PUSH_MARGIN
    push_target_x = min(1.50, obs_x + PUSH_EXTRA)
    print(f"\nObs0 push RIGHT: approach_x={approach_x:.3f}, push_target={push_target_x:.3f}")

print(f"PUSH_Y={PUSH_Y_val}, NAV_HIGH_Y={NAV_HIGH_Y}")

# Now simulate the push approach - run the actual approach for a bit
from motion_planning import BiRRT
primitives = {'BiRRT': BiRRT}
from approach import GeneratedApproach
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

from obs_helpers import any_obstruction_on_surface, is_holding_block, block_on_surface
print("\n--- Simulating steps ---")
for step in range(300):
    action = approach.get_action(obs)
    r = get_robot(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if step % 50 == 0 or terminated or truncated:
        r2 = get_robot(obs)
        any_obs = any_obstruction_on_surface(obs)
        holding = is_holding_block(obs)
        on_surf = block_on_surface(obs)
        o0 = get_obstruction(obs, 0)
        print(f"step={step}: robot=({r2['x']:.3f},{r2['y']:.3f}) theta={r2['theta']:.2f} "
              f"obs0=({o0['x']:.3f},{o0['y']:.3f}) any_obs={any_obs} holding={holding} on_surf={on_surf} "
              f"action=({action[0]:.3f},{action[1]:.3f}) done={terminated}")
    if terminated or truncated:
        print(f"Terminated at step {step}")
        break
