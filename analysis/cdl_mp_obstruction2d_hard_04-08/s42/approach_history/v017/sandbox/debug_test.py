import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}

for seed in [42]:
    obs, info = env.reset(seed=seed)
    approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
    approach.reset(obs, info)

    done = False
    steps = 0
    max_steps = 600
    total_reward = 0.0
    last_phase = None

    while not done and steps < max_steps:
        # Print phase changes
        cur = approach._current
        phase = getattr(cur, '_phase', '?')
        cls = cur.__class__.__name__
        key = (cls, phase)
        if key != last_phase:
            print(f"  step={steps} [{cls}] phase={phase}")
            last_phase = key

        action = approach.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    print(f"Seed {seed}: steps={steps}, reward={total_reward:.0f}, done={done}, terminated={terminated}")
