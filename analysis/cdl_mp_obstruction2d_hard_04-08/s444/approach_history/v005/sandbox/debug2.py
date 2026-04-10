import numpy as np
import sys
sys.path.insert(0, '/sandbox')
sys.path.insert(0, '/sandbox/primitives')
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import get_robot, get_obstruction, any_obstruction_on_surface
from act_helpers import clip_action

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

prev_pos = None
for step in range(30):
    robot = get_robot(obs)
    pos = (robot['x'], robot['y'])
    action = approach.get_action(obs)
    sub = approach._current._current
    sub_phase = getattr(sub, '_phase', '?') if sub else '?'
    stuck = prev_pos and abs(pos[0]-prev_pos[0])<0.001 and abs(pos[1]-prev_pos[1])<0.001
    if step >= 8:
        print(f"step {step:3d}: ({pos[0]:.4f},{pos[1]:.4f}) arm={robot['arm_joint']:.3f} theta={robot['theta']:.3f} phase={sub_phase} action=({action[0]:.3f},{action[1]:.3f},{action[2]:.3f},{action[3]:.3f}) {'STUCK' if stuck else ''}")
    prev_pos = pos
    obs, reward, terminated, truncated, info = env.step(action)
