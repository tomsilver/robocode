"""Test approach."""
import sys, os
sys.path.insert(0, '/sandbox')
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from primitives.motion_planning import BiRRT

env = KinderGeom2DEnv('kinder/Obstruction2D-o4-v0')
primitives = {'BiRRT': BiRRT}

# Test basic usage
obs, info = env.reset(seed=42)
print('obs shape:', obs.shape)
print('BiRRT class:', BiRRT)

# Test if BiRRT can be instantiated
rng = np.random.default_rng(0)
sample_fn = lambda s: np.array([rng.uniform(0, 1.6), rng.uniform(0.2, 1.0)])
extend_fn = lambda s1, s2: [s1 + (s2-s1)*t for t in np.linspace(0.1, 1, 10)]
collision_fn = lambda s: False
distance_fn = lambda s1, s2: np.linalg.norm(s1 - s2)
birrt = BiRRT(sample_fn, extend_fn, collision_fn, distance_fn, rng, 5, 200, 10)
path = birrt.query(np.array([0.5, 0.5]), np.array([1.0, 0.8]))
print('Path found:', len(path) if path else None)
print('Test passed!')
