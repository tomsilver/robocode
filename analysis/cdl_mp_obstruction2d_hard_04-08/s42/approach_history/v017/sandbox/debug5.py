import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, is_holding, NUM_OBSTRUCTIONS, is_obstruction_blocking_surface

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}

obs, info = env.reset(seed=42)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

done = False
steps = 0

while not done and steps < 120:
    cur = approach._current
    phase = getattr(cur, '_phase', '?')
    if steps % 5 == 0 or phase in ['GRASP', 'RELEASE', 'FIND']:
        blocking = [i for i in range(NUM_OBSTRUCTIONS) if is_obstruction_blocking_surface(obs, i)]
        print(f"step={steps} phase={phase} holding={is_holding(obs)} blocking={blocking}")
        for i in range(NUM_OBSTRUCTIONS):
            ob = extract_obstruction(obs, i)
            print(f"  obs{i}: x={ob['x']:.3f} y={ob['y']:.3f} top={ob['y']+ob['height']:.3f}")

    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    steps += 1
