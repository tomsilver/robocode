import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, NUM_OBSTRUCTIONS

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}

obs, info = env.reset(seed=1)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)
done = False; steps = 0

while not done and steps < 200:
    cur = approach._current
    phase = getattr(cur, '_phase', '?')
    r = extract_robot(obs)
    if phase == 'NAV_HZDROP' and steps > 85 and steps == 130:
        print(f"step={steps} robot=({r['x']:.3f},{r['y']:.3f}) arm_joint={r['arm_joint']:.3f}")
        for i in range(NUM_OBSTRUCTIONS):
            ob = extract_obstruction(obs, i)
            print(f"  obs{i}: x=[{ob['x']:.3f},{ob['x']+ob['width']:.3f}] y=[{ob['y']:.3f},{ob['y']+ob['height']:.3f}]")
        print(f"  drop_x={cur._drop_x:.3f} target_idx={cur._target_idx}")
        # carried obs
        ob = extract_obstruction(obs, cur._target_idx)
        print(f"  carried: w={ob['width']:.3f} h={ob['height']:.3f}")
        carry_top = r['y'] - 0.215
        print(f"  carried top={carry_top:.3f} bottom={carry_top-ob['height']:.3f} x=[{r['x']-ob['width']/2:.3f},{r['x']+ob['width']/2:.3f}]")
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    steps += 1
print(f"done={done}")
