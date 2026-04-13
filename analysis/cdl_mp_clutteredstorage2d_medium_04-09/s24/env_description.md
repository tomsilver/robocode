# kinder/ClutteredStorage2D-b3-v0

A 2D environment where the goal is to put all blocks inside a shelf.

The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. Objects can be grasped and ungrasped when the end effector makes contact.


## Variant

This variant has 3 blocks (1 initially in the shelf, 2 initially outside).

## Observation Space

The entries of an array in this Box space correspond to the following object features:
| **Index** | **Object** | **Feature** |
| --- | --- | --- |
| 0 | robot | x |
| 1 | robot | y |
| 2 | robot | theta |
| 3 | robot | base_radius |
| 4 | robot | arm_joint |
| 5 | robot | arm_length |
| 6 | robot | vacuum |
| 7 | robot | gripper_height |
| 8 | robot | gripper_width |
| 9 | shelf | x |
| 10 | shelf | y |
| 11 | shelf | theta |
| 12 | shelf | static |
| 13 | shelf | color_r |
| 14 | shelf | color_g |
| 15 | shelf | color_b |
| 16 | shelf | z_order |
| 17 | shelf | width |
| 18 | shelf | height |
| 19 | shelf | x1 |
| 20 | shelf | y1 |
| 21 | shelf | theta1 |
| 22 | shelf | width1 |
| 23 | shelf | height1 |
| 24 | shelf | z_order1 |
| 25 | shelf | color_r1 |
| 26 | shelf | color_g1 |
| 27 | shelf | color_b1 |
| 28 | block0 | x |
| 29 | block0 | y |
| 30 | block0 | theta |
| 31 | block0 | static |
| 32 | block0 | color_r |
| 33 | block0 | color_g |
| 34 | block0 | color_b |
| 35 | block0 | z_order |
| 36 | block0 | width |
| 37 | block0 | height |
| 38 | block1 | x |
| 39 | block1 | y |
| 40 | block1 | theta |
| 41 | block1 | static |
| 42 | block1 | color_r |
| 43 | block1 | color_g |
| 44 | block1 | color_b |
| 45 | block1 | z_order |
| 46 | block1 | width |
| 47 | block1 | height |
| 48 | block2 | x |
| 49 | block2 | y |
| 50 | block2 | theta |
| 51 | block2 | static |
| 52 | block2 | color_r |
| 53 | block2 | color_g |
| 54 | block2 | color_b |
| 55 | block2 | z_order |
| 56 | block2 | width |
| 57 | block2 | height |


## Action Space

The entries of an array in this Box space correspond to the following action features:
| **Index** | **Feature** | **Description** | **Min** | **Max** |
| --- | --- | --- | --- | --- |
| 0 | dx | Change in robot x position (positive is right) | -0.050 | 0.050 |
| 1 | dy | Change in robot y position (positive is up) | -0.050 | 0.050 |
| 2 | dtheta | Change in robot angle in radians (positive is ccw) | -0.196 | 0.196 |
| 3 | darm | Change in robot arm length (positive is out) | -0.100 | 0.100 |
| 4 | vac | Directly sets the vacuum (0.0 is off, 1.0 is on) | 0.000 | 1.000 |


## Reward

A penalty of -1.0 is given at every time step until termination, which occurs when all blocks are inside the shelf.


## Example Usage

```python
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv

env = KinderGeom2DEnv("kinder/ClutteredStorage2D-b3-v0")
obs, info = env.reset(seed=0)
print(obs.shape)  # (58,)

# Take a random action
action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)

# Save and restore state
saved = env.get_state()
env.step(env.action_space.sample())
env.set_state(saved)  # restores to the saved state

# Run an episode
obs, info = env.reset(seed=1)
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

`obs` and `action` are numpy arrays matching the tables above.

## Source Code

`KinderGeom2DEnv` is a thin wrapper. The underlying environment logic lives in the `kinder` package. To find the source files:

```python
import kinder.envs.geom2d
print(kinder.envs.geom2d.__path__)
```

Key files in that directory:
- `base_env.py` — `step()` transition dynamics and collision handling
- The environment-specific module (e.g. `motion2d.py`) — reward function (`_get_reward_and_done`), config, and scene generation
- `object_types.py` — object type definitions and feature names