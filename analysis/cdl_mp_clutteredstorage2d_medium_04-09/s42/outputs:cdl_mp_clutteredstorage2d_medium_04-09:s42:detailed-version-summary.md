## outputs/cdl_mp_clutteredstorage2d_medium_04-09/s42: detailed version summary



### v00

initial, not distinguish the blocks in shelf and other blocks,but accidently go to the block in shelf first.

### v01
this upgrades pickup geometry.

What changed:

- integrated `_find_approach()` into `PickBlock`
- pickup can now choose among multiple approach directions instead of only one
- stores the chosen approach theta for later rotation

Key code:

```python
goal_x, goal_y, self._approach_theta = _find_approach(block, other_blocks)
birrt = _make_birrt(self._primitives, obs, other_block_names, self._rng)
```

Main failure in this version:

- not picking the block in shelf correctly,pickup/placement still assumes the held block sits exactly at the gripper origin


### v02

not fix the bug


### v03

 this is a grasp/placement offset fix.

What changed:

- corrected suction offset in pickup termination
- added `arm_offset` and `perp_offset` for placement navigation

Main failure in this version:

- not pick the block i then transport



### v04

not fixing the right bug( not pick the block i then transport)

### v05

not fixing the right bug( not pick the block i then transport)

### v06

not fixing the right bug( not pick the block i then transport)




### v07

not fixing the right bug( not pick the block i then transport)


### v08

not fixing the right bug( not pick the block i then transport)

### v09

not fixing the right bug( not pick the block i then transport)


### v10

this tightens approach-pose validity,which fix the bug finally

What changed:

- `_find_approach()` now checks that the clipped robot position can still physically reach the block

Key code:

```python
reach_x = gx + arm_dist * math.cos(theta)
reach_y = gy + arm_dist * math.sin(theta)
if abs(reach_x - bx) > 0.15 or abs(reach_y - by) > 0.15:
    continue
```

Main failure in this version:

- navigation failure when holding blocks



### v11

switch back to not clean the shelf first.
    


### v12

this improves pickup safety.

What changed:

- BiRRT now avoids the target block during navigation
- rotation is forced CCW to reduce arm sweep through nearby blocks

Main failure in this version:

- navigation failure and not clean the shelf first



### v13

not fix the bug

### v14

High-level rewrite.

What changed:

- added a top-down slot strategy
- if `block0` starts in the shelf, move it out first
- place outside blocks into high-to-low slots
- place `block0` last at the bottom
- introduced `MoveBlockToTemp`

Key code:

```python
if outside and block0_in_shelf:
    pick0 = PickBlock('block0', self._primitives, place_slot=99, allow_in_shelf=True)
    move0 = MoveBlockToTemp('block0', self._primitives, temp_x=2.5, temp_y=1.0)
    self._behaviors.extend([pick0, move0])
    for slot, block_name in enumerate(outside):
        pick = PickBlock(block_name, self._primitives, place_slot=slot)
        place = PlaceInShelf(block_name, self._primitives, slot_index=slot)
        self._behaviors.extend([pick, place])
```

Main failure in this version:

- pick up the block in shelf failure


### v15

not fix the bug


### v16


not fix the bug


### v17

not fix the bug


### Overall evolution



