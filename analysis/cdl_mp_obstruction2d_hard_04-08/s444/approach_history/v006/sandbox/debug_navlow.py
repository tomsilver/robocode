"""Debug why nav_low gets stuck."""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import get_robot, get_obstruction
from behaviors import PickOneObstruction, OBS_PICK_ARM, OBS_PICK_CLEARANCE
from act_helpers import NAV_HIGH_Y, K_POS, MAX_DX, MAX_DY
from act_helpers import normalize_angle, K_THETA, MAX_DTHETA
import numpy as np

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, _ = env.reset(seed=0)
o0 = get_obstruction(obs, 0)
obs_x = o0['x']; obs_top = o0['y'] + o0['height']/2
pick_y = obs_top + OBS_PICK_ARM + OBS_PICK_CLEARANCE
print(f"obs_x={obs_x:.4f} obs_top={obs_top:.4f} pick_y={pick_y:.4f}")

b = PickOneObstruction(None, 0)
b.reset(obs)

prev_y = None
for step in range(300):
    r = get_robot(obs)
    action = b.step(obs)
    obs, _, _, _, _ = env.step(action)
    r2 = get_robot(obs)
    
    if b._phase == 'nav_low' and step < 200:
        if prev_y is None or r2['y'] != prev_y or step % 10 == 0:
            print(f"  step={step:3d} ph={b._phase} action=[{action[0]:.3f},{action[1]:.3f}] "
                  f"x:{r['x']:.4f}→{r2['x']:.4f} y:{r['y']:.4f}→{r2['y']:.4f} arm:{r2['arm_joint']:.4f}")
        prev_y = r2['y']
    
    if b.terminated(obs):
        print(f"CLEARED at step {step}!")
        break
