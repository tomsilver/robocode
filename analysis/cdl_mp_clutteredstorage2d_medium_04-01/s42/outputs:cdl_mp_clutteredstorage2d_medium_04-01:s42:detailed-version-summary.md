## outputs/cdl_mp_clutteredstorage2d_medium_04-01/s42: detailed version summary

### v00

First real policy: a thin queue of `PickBlockBehavior` then `PlaceBlockBehavior` for each outside block.

What changed:

- created the initial direct `pick -> place` queue
- added BiRRT-based base navigation

Main failure in this version:

- the policy assumes every outside block can be directly picked and directly inserted
- it has no mechanism for the initial shelf occupant, so the shelf stays structurally blocked



### v01

Low-level grasp refinement.

What changed:

- reduced pick arm extension step size so the suction phase does not jump through the narrow successful contact window

Key code:

```python
DARM_SMALL = 0.02   # small arm step for precise extension
```

Main failure in this version:

- did not fix the grasp failure
- direct placement still runs into the occupied shelf




### v02

A major strategy rewrite.

What changed:

- replaced the plain `pick -> place` queue with a staged rearrangement story
- introduced `TempDropBehavior`
- introduced `HoldInShelfBehavior`
- changed pick logic to compute a reorienting pick angle

Key code:

```python
self._behaviors.append(PickBlockBehavior(shelf_block, self._primitives))
self._behaviors.append(TempDropBehavior(self._primitives))
for i, bn in enumerate(outside):
    self._behaviors.append(PickBlockBehavior(bn, self._primitives))
    self._behaviors.append(PlaceBlockBehavior(self._primitives, deposit_arm_joint=arm_j))
self._behaviors.append(PickBlockBehavior(shelf_block, self._primitives))
self._behaviors.append(HoldInShelfBehavior(self._primitives))
```

And the key geometric idea:

```python
def compute_pick_angle(block_theta: float) -> float:
    alpha_k0 = _norm(block_theta + math.pi / 2)
    alpha_k1 = _norm(block_theta - math.pi / 2)
    if math.sin(alpha_k0) > 0:
        return alpha_k0
    return alpha_k1
```

New strategy is closer to oracle,but motion planning when holding object is still causing a lot of failures.




### v03

 tries to make transport safer when already holding a block.

What changed:

- added a lower navigation ceiling while carrying a block
- lowered temp-drop navigation height
- simplified the temp-drop path shape

Key code:

```python
HOLDING_NAV_Y_CEILING = 2.25
TEMP_DROP_Y_NAV = 0.40
```

Main failure in this version:

- the new constraints overcorrect
- the robot avoids one collision mode but loses maneuvering flexibility and still gets trapped after release 






### v04

this adds post-drop recovery.

What changed:

- added an `_ESCAPE` phase to `TempDropBehavior`
- after releasing the temp block, the robot moves away laterally instead of stopping under it

Key code:

```python
elif self._phase == _RELEASE:
    self._escape_x = (self._target_robot_x - 2.0
                      if self._target_robot_x > 2.5
                      else self._target_robot_x + 2.0)
    self._phase = _ESCAPE
    return build_action(vacuum=0.0)

elif self._phase == _ESCAPE:
    dx = np.clip(self._escape_x - robot.x, -DX_LIM, DX_LIM)
    if abs(self._escape_x - robot.x) < POS_TOL:
        self._phase = _DONE
```

Main failure in this version:

- drop is still fail,collide with other block while transporting




### v05

placement-geometry correction.

What changed:

- changed x clamping so the robot uses `ROBOT_RADIUS` rather than an extra arbitrary margin
- the intent was to let the carried block actually fit within the shelf inner x-bounds

Main failure in this version:

- drop is still fail,collide with other block while transporting


### v06

Compared with `v05`, this is the first explicit floor-clutter avoidance upgrade.

What changed:

- added `_floor_block_obstacles(...)`
- began passing floor blocks as obstacles into BiRRT path planning

Key code:

```python
def _floor_block_obstacles(obs: np.ndarray, exclude_names: list[str]) -> list[tuple]:
    obstacles = []
    for bn in BLOCK_NAMES:
        if bn in exclude_names:
            continue
        if is_block_in_shelf(obs, bn):
            continue
        rect = extract_rect(obs, bn)
        cx, cy = block_center(rect)
        obstacles.append((cx, cy, BLOCK_HALF_DIAG + margin))
    return obstacles
```

Main failure in this version:

- obstacle modeling is still inconsistent across behaviors
- the target block is mishandled during approach, and `TempDropBehavior` still relies on a weaker motion story




### v07

this finally find the real problem and fixes two specific planner mistakes.

What changed:

- the target block is no longer silently excluded from obstacle handling during approach
- `TempDropBehavior` now also uses BiRRT rather than only a simplistic hand-authored path

Main failure in this version:

- global path feasibility improves, but final shelf insertion still suffers from lateral misalignment




### v08

this is a lateral alignment correction during place / hold-in-shelf.

What changed:

- adjusted robot x during placement based on measured block center rather than only nominal offsets

Main failure in this version:

- this version is already locally strong
- the remaining issue is not a single exposed bug but overall brittleness of the `TempDrop + HoldInShelf` script




### v09

this is mostly an auto-commit snapshot.



### v10

 a broken reset and runnning error

What changed:

- `act_helpers.py` and `obs_helpers.py` were reset as part of sandbox reinitialization
- the exported version does not load correctly because helper symbols are missing



### v11

New strategy line.

What changed:

- discarded `TempDrop + HoldInShelf`
- went back to a much cleaner direct queue: `pick outside blocks -> place outside blocks`
- introduced `AllDoneBehavior`
- added “skip pick if already holding this block” logic

Key code:

```python
outside = get_outside_block_indices(state)
for block_idx in outside:
    if is_holding_block(state, block_idx):
        self._behaviors.append(PlaceBlockBehavior(block_idx, self._primitives))
    else:
        self._behaviors.append(PickBlockBehavior(block_idx, self._primitives))
        self._behaviors.append(PlaceBlockBehavior(block_idx, self._primitives))
```

Main failure in this version:

- the queue is simpler, but all geometry is still semantically wrong
- block `x/y` is treated as center-like data, so grasp and placement geometry are systematically off



### v12

geometry-semantics repair.

What changed:

- corrected the interpretation of block `x/y` as rectangle-corner coordinates rather than center coordinates
- fixed `block_vertices`
- fixed `block_center`
- fixed shelf containment and holding heuristics to use the corrected geometry

Key code:

```python
def block_vertices(block: BlockPose):
    local_corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    ...

def block_center(block: BlockPose) -> tuple[float, float]:
    cx = x + cos_t * w / 2 - sin_t * h / 2
    cy = y + sin_t * w / 2 + cos_t * h / 2
    return cx, cy
```

Main failure in this version:

- geometry semantics are now correct, but pick targeting is still wrong
- the robot still aims too generically, often near the block center, and can terminate pickup too early




### v13

not fix the bug


### v14

not fixing the right bug 

### v15

not fixing the right bug

### v16

this adds the second line's most important task-specific hack.

What changed:

- added `MoveBlock0UpBehavior`
- inserted it at the front of the behavior queue when `block0` is low in the shelf

Key code:

```python
mover = MoveBlock0UpBehavior(self._primitives)
if mover.initializable(state):
    self._behaviors.append(mover)
```

Main failure in this version:

- the high-level bottleneck is now explicitly handled
- the remaining issue is local grasp-face choice and obstacle-aware approach planning


### v17

this refines face-choice scoring and pickup navigation.

What changed:

- changed face scoring to prefer faces closer to the current robot position
- added block obstacle avoidance to BiRRT during pickup planning

Main failure in this version:

- local reachability is improved, but the face-choice objective is wrong
- the policy chooses what is easiest to grab now, not what will be easiest to place later


### v18

this changes the face-choice objective again.

What changed:

- switched from “closest face to robot” to “best orientation for horizontal placement”

Main failure in this version:

- the face objective is better, but approach planning can still exploit false-feasible routes that cut too close to the target block



### v19

this tightens obstacle modeling during approach.

What changed:

- the target block itself is included in BiRRT obstacle construction so the base routes around it rather than clipping toward it unrealistically

Main failure in this version:

- geometry and routing are much cleaner, but grasp-state detection is still too generous
- the robot can think it is holding a block just because the gripper is near it



### v20

grasp-state fix.

What changed:

- tightened the holding heuristic threshold from generous values down to `0.20`
- specifically targeted false-positive grasp detection

Key code:

```python
HOLDING_DIST_THRESHOLD = 0.20
```





### Overall evolution


The final policy is much better than the initial one because it eventually combines:

- correct geometry semantics
- task-aware face-normal grasping
- stricter pickup / placement termination
- explicit handling of low `block0`
- tighter obstacle modeling
- stricter held-block detection

The remaining ceiling is also clear from the version history:

- many local bugs are eventually solved
- but the controller still commits to a fixed queue at reset
- it never learns oracle-style online reselection of `compact / clear / store`
