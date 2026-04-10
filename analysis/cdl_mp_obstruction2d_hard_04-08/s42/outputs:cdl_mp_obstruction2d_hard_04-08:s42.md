## outputs/cdl_mp_obstruction2d_hard_04-08/s42 (0.72)



![image-20260410110459290](/Users/libowen/Library/Application Support/typora-user-images/image-20260410110459290.png)

#### v00

Has running bug



#### v01

Fix bottom-left vs center coordinate confusion for all rectangles

```
def approach_xy_for_pick(obj: dict, arm_length: float) -> tuple:
    """Compute robot (x, y) for approaching the object from above (theta=-pi/2).

    obs (x, y) is the BOTTOM-LEFT corner of the object.
    Object center: (x + w/2, y + h/2).
    When at (center_x, center_y + arm_length) with arm fully extended (theta=-pi/2),
    gripper tip is at object center.
    """
    center_x = obj['x'] + obj['width'] / 2.0
    center_y = obj['y'] + obj['height'] / 2.0
    return center_x, center_y + arm_length
```

This y is wrong, and robot will collide with the obstruction.



#### v02

Fix approach_xy_for_pick: use obj_top + arm_length + 0.010

This fixed the previous problem, but has a similar "early terminate" issue:

```
    def terminated(self, obs) -> bool:
        return not any_obstruction_on_surface(obs)
```

And the on_surface predicate is wrong:

```
    for vx in [bl_x, bl_x + rw]:
        if not _rect_contains_point_bl(surf['x'], surf['y'],
                                       surf['width'], surf['height'],
                                       vx, offset_y):
            return False
    return True
```

This means fully contain (only when both bottom vertices are in, we count as on)



#### v03

![image-20260410111932638](/Users/libowen/Library/Application Support/typora-user-images/image-20260410111932638.png)

Fix behavior termination and carry navigation

This is better, but the on obstruction condition is still wrong



#### v04

Use broader surface-blocking check to clear adjacent obstructions

![image-20260410112326267](/Users/libowen/Library/Application Support/typora-user-images/image-20260410112326267.png)

It first picks the lower obstruction, so it can't extend arm



#### v05

Pick tallest obstruction first to avoid lateral gripper collision

Worked!



#### v06

Add descent-before-release phases to avoid obstructions floating at gripper height

![image-20260410112616565](/Users/libowen/Library/Application Support/typora-user-images/image-20260410112616565.png)



#### v07

Simplify ClearAllObstructions: release straight up, no horizontal carry (avoids arm/gripper collision with dropped obstructions)

![image-20260410112649041](/Users/libowen/Library/Application Support/typora-user-images/image-20260410112649041.png)

Does not work.





#### v08

Add carry-aware BiRRT collision fn and smart drop zones to avoid floating obstruction collisions

Success!



#### v09

Fix _choose_drop_x: more candidates, correct target block exclusion, better fallback



#### v10

Fix terminated() threshold to 0.25; use carry-aware path in PlaceTargetBlock



#### v11

PlaceTargetBlock: carry-aware surf_x selection and path planning; base path for descent





#### Summary of final failures

Similar to no_mp 0.82

wait what, isn't it already using motion planning?

<img src="/Users/libowen/Library/Application Support/typora-user-images/image-20260410130657744.png" alt="image-20260410130657744" style="zoom:50%;" />

<img src="/Users/libowen/Library/Application Support/typora-user-images/image-20260410131014765.png" alt="image-20260410131014765" style="zoom:50%;" />

```
        if self._phase == _PLAN_HIGH:
            self._update_targets(obs)
            obs_rects = self._obs_rects_for_carry(obs)
            # Rise to NAV_HEIGHT while carrying block
            path = plan_carry_path(obs, self._primitives, rx, NAV_HEIGHT,
                                   carry_w=carry_w, carry_h=carry_h,
                                   obstacle_rects=obs_rects)
            self._path = path or [np.array([rx, NAV_HEIGHT])]
```

Nav_high has not collision free path; it will just take a direct connecting path...

```
            obs_rects = self._obs_rects_for_carry(obs)
            # Fly horizontally to surf_x at NAV_HEIGHT, carry-aware
            path = plan_carry_path(obs, self._primitives, self._surf_x, NAV_HEIGHT,
                                   carry_w=carry_w, carry_h=carry_h,
                                   obstacle_rects=obs_rects)
            self._path = path or [np.array([self._surf_x, NAV_HEIGHT])]
```

No collision free path it will just use direct connect path..

<img src="/Users/libowen/Library/Application Support/typora-user-images/image-20260410132620318.png" alt="image-20260410132620318" style="zoom:50%;" />



#### Some thoughts

Claude Code seems to be pretty good at debugging a specific seed if:

- Each fix are relatively independent, e.g., changing one thing does not make another thing unfixable.
  - E.g., termination issue, pick order, rectangle bugs are independent, fixing any is good overall.
- When it moves to multiple seeds, things become harder:
  - It will try to find "local solutions", e.g., some specific valuable regions for placing blocks, instead of a global solution, e.g., use CSP solver/sampling to find a collision-free zone.
  - These "local solutions" do not work for all configurations in a domain, and the performance actually depends on them.