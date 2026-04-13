import sys, os
sys.path.insert(0, '/sandbox')
os.chdir('/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from obs_helpers import SHELF_BASE_IDX, SHELF_FEATURES
from primitives.motion_planning import BiRRT

env = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs, info = env.reset(seed=0)

base = SHELF_BASE_IDX
for i, feat in enumerate(SHELF_FEATURES):
    print(f"  {feat} = {obs[base+i]:.4f}")

from obs_helpers import BLOCK0_BASE_IDX, RECT_FEATURES
base = BLOCK0_BASE_IDX
for i, feat in enumerate(RECT_FEATURES):
    print(f"  block0.{feat} = {obs[base+i]:.4f}")

# Also simulate placing a block and see where it ends up
print("\n--- Testing placement ---")
from approach import GeneratedApproach
from primitives.motion_planning import BiRRT
PRIMITIVES = {'BiRRT': BiRRT}

env2 = KinderGeom2DEnv('kinder/ClutteredStorage2D-b3-v0')
obs2, info2 = env2.reset(seed=0)
approach = GeneratedApproach(env2.action_space, env2.observation_space, PRIMITIVES)
approach.reset(obs2, info2)

# Run until first block placed
for step in range(200):
    action = approach.get_action(obs2)
    obs2, reward, terminated, truncated, info2 = env2.step(action)
    cur = approach._current
    ph = getattr(cur, '_phase', None)
    if type(cur).__name__ == 'PickupBlock' and step > 50:
        # block1 should be placed
        from obs_helpers import extract_rect, BLOCK_NAMES, is_block_in_shelf, get_block_center, BLOCK1_BASE_IDX
        b1 = extract_rect(obs2, 'block1')
        cx, cy = get_block_center(obs2, 'block1')
        in_shelf = is_block_in_shelf(obs2, 'block1')
        print(f"  After s={step}: block1 corner=({b1.x:.3f},{b1.y:.3f}) theta={b1.theta:.3f} center=({cx:.3f},{cy:.3f}) in={in_shelf}")
        print(f"  block1 static={obs2[BLOCK1_BASE_IDX+3]:.1f}")
        break

