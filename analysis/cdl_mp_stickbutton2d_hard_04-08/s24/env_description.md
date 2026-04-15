# kinder/StickButton2D-b5-v0

A 2D environment where the goal is to touch all buttons, possibly by using a stick for buttons that are out of the robot's direct reach.

In this environment, there are always 5 buttons.

The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector.


## Variant

This variant has 5 buttons to press.

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
| 9 | stick | x |
| 10 | stick | y |
| 11 | stick | theta |
| 12 | stick | static |
| 13 | stick | color_r |
| 14 | stick | color_g |
| 15 | stick | color_b |
| 16 | stick | z_order |
| 17 | stick | width |
| 18 | stick | height |
| 19 | button0 | x |
| 20 | button0 | y |
| 21 | button0 | theta |
| 22 | button0 | static |
| 23 | button0 | color_r |
| 24 | button0 | color_g |
| 25 | button0 | color_b |
| 26 | button0 | z_order |
| 27 | button0 | radius |
| 28 | button1 | x |
| 29 | button1 | y |
| 30 | button1 | theta |
| 31 | button1 | static |
| 32 | button1 | color_r |
| 33 | button1 | color_g |
| 34 | button1 | color_b |
| 35 | button1 | z_order |
| 36 | button1 | radius |
| 37 | button2 | x |
| 38 | button2 | y |
| 39 | button2 | theta |
| 40 | button2 | static |
| 41 | button2 | color_r |
| 42 | button2 | color_g |
| 43 | button2 | color_b |
| 44 | button2 | z_order |
| 45 | button2 | radius |
| 46 | button3 | x |
| 47 | button3 | y |
| 48 | button3 | theta |
| 49 | button3 | static |
| 50 | button3 | color_r |
| 51 | button3 | color_g |
| 52 | button3 | color_b |
| 53 | button3 | z_order |
| 54 | button3 | radius |
| 55 | button4 | x |
| 56 | button4 | y |
| 57 | button4 | theta |
| 58 | button4 | static |
| 59 | button4 | color_r |
| 60 | button4 | color_g |
| 61 | button4 | color_b |
| 62 | button4 | z_order |
| 63 | button4 | radius |


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

A penalty of -1.0 is given at every time step until all buttons have been pressed (termination).


## Example Usage

```python
import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv

env = KinderGeom2DEnv("kinder/StickButton2D-b5-v0")
obs, info = env.reset(seed=0)
print(obs.shape)  # (64,)

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