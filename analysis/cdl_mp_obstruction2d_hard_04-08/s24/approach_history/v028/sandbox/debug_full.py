"""Full trace of seed 0 up to 300 steps."""
import sys
sys.path.insert(0, "primitives")
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from approach import GeneratedApproach
from obs_helpers import extract_robot, extract_obstruction, extract_surface, extract_block, NUM_OBSTRUCTIONS

primitives = {"BiRRT": BiRRT}
env = KinderGeom2DEnv("kinder/Obstruction2D-o4-v0")

obs, info = env.reset(seed=0)
approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

print("Initial behaviors:", [type(b).__name__ + (f"[{b._i}]" if hasattr(b,'_i') else "") for b in list(approach._behaviors)])
print("Current:", type(approach._current).__name__ + (f"[{approach._current._i}]" if hasattr(approach._current,'_i') else ""))

prev_phase = None
prev_class = None

for step in range(300):
    cur = approach._current
    cur_class = type(cur).__name__ + (f"[{cur._i}]" if hasattr(cur,'_i') else "")
    cur_phase = cur._phase

    if cur_class != prev_class or cur_phase != prev_phase:
        r = extract_robot(obs)
        print(f"step={step} {cur_class}.{cur_phase} robot=({r['x']:.3f},{r['y']:.3f}) theta={r['theta']:.3f} arm={r['arm_joint']:.3f} vac={r['vacuum']:.0f}")
        prev_phase = cur_phase
        prev_class = cur_class

    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        r = extract_robot(obs)
        print(f"DONE at step={step+1} reward={reward} terminated={terminated}")
        break

r = extract_robot(obs)
blk = extract_block(obs)
surf = extract_surface(obs)
print(f"\nFinal: robot=({r['x']:.3f},{r['y']:.3f}) theta={r['theta']:.3f} arm={r['arm_joint']:.3f} vac={r['vacuum']:.0f}")
print(f"Block: ({blk['cx']:.3f},{blk['cy']:.3f}) on_surf={blk['y1']:.3f}~{surf['y2']:.3f}")
print(f"Current behavior: {type(approach._current).__name__}.{approach._current._phase}")

env.close()
