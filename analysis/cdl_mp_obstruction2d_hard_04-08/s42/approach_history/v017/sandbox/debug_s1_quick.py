import sys; sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}
obs, info = env.reset(seed=1)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

phase_counts = {}
for step in range(2000):
    # Track current behavior and phase
    cur = approach._current
    bname = type(cur).__name__
    ph = getattr(cur, '_phase', '?')
    key = (bname, ph)
    phase_counts[key] = phase_counts.get(key, 0) + 1
    
    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Done at step {step+1}")
        break

# Show top phases by count
sorted_phases = sorted(phase_counts.items(), key=lambda x: -x[1])
print("Top phase counts:")
for (bname, ph), cnt in sorted_phases[:15]:
    print(f"  {bname}.{ph}: {cnt}")
