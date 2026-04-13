"""Debug full pipeline for one seed."""
import sys
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT
from obs_helpers import (
    extract_robot, extract_block, extract_shelf_inner, block_center, block_vertices,
    is_block_in_shelf, is_holding_block, SHELF_FLOOR_Y
)
from approach import GeneratedApproach

primitives = {'BiRRT': BiRRT}
env = KinderGeom2DEnv("kinder/ClutteredStorage2D-b3-v0")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
obs, info = env.reset(seed=seed)

approach = GeneratedApproach(env.action_space, env.observation_space, primitives)
approach.reset(obs, info)

print(f"Seed {seed}")
shelf = extract_shelf_inner(obs)
print(f"Shelf: y=[{shelf.y:.3f},{shelf.y_max:.3f}]")
for i in range(3):
    b = extract_block(obs, i)
    bcy = block_center(b)[1]
    print(f"Block{i}: center_y={bcy:.4f}, in_shelf={is_block_in_shelf(obs, i)}")
print()

done = False
step = 0
last_behavior_name = None
max_steps = 2000

while not done and step < max_steps:
    # Track behavior changes
    bname = type(approach._current).__name__
    if bname != last_behavior_name:
        robot = extract_robot(obs)
        print(f"  step={step}: Starting {bname}, robot=({robot.x:.3f},{robot.y:.3f})")
        for i in range(3):
            b = extract_block(obs, i)
            bcy = block_center(b)[1]
            print(f"    Block{i}: cy={bcy:.4f}, in_shelf={is_block_in_shelf(obs, i)}, held={is_holding_block(obs, i)}")
        last_behavior_name = bname

    action = approach.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    step += 1

print(f"\nFinal: step={step}, done={done}")
for i in range(3):
    b = extract_block(obs, i)
    bcy = block_center(b)[1]
    verts = block_vertices(b)
    miny = min(vy for _, vy in verts)
    maxy = max(vy for _, vy in verts)
    print(f"Block{i}: cy={bcy:.4f}, y=[{miny:.4f},{maxy:.4f}], in_shelf={is_block_in_shelf(obs, i)}")

env.close()
