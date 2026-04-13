## outputs/cdl_mp_clutteredstorage2d_medium_04-09/s24: detailed version summary




### v00

first functioning policy.


Main failure in this version:

- wrong grasping

### v01

not fixing the right bug,this changes the transport style during placement,

What changed:

- removed the explicit retract phase from `PlaceBlock`
- planned navigation at reset
- kept the arm extended while navigating to avoid dropping the held block

Main failure in this version:

- still wrong grasping

### v02

not fixing the grasping the bug,this is a queue-control fix.

What changed:

- fixed `NAVIGATE` fallthrough by using a `while` loop to skip close waypoints
- returned `VAC_ON` during carrying navigation

Key code:

```python
while self._path_idx < len(self._path):
    wp = self._path[self._path_idx]
    ...
    if math.sqrt(dx ** 2 + dy ** 2) < XY_TOL:
        self._path_idx += 1
    else:
        break
```

Main failure in this version:

- still wrong grasping




### v03

still not fix the grasping bug

What changed:

- tightened ORIENT position/theta tolerances
- started servoing x and y during EXTEND so the block stays centered in the shelf slot

Main failure in this version:

- grasp failure



### v04

not fix the bug

### v05

not fix the bug


### v06

finally grasp the block

What changed:

- fixed false pickup termination
- added navigation stuck detection
- made `PlaceBlock` skip immediately if nothing is actually held
- added arm stuck detection

Key code family:

```python
if min_dist > 0.4:
    self._phase = self._PH_DONE
    return
```

Main failure in this version:


- the shelf is not clear




### v07

not cleanning shelf first



### v08

not clear shelf first.


### v09

high-level strategy change in the whole run, clear shelf first

What changed:

- added `allow_shelf=True` so `block0` can be picked from inside the shelf
- introduced `DropBlock`
- changed `approach.py` to: remove `block0`, place `block1`, place `block2`, then place `block0`
- made `PlaceBlock` rely on pushing rather than early “in shelf” exit

Key code:

```python
if is_block_in_shelf(obs, 'block0'):
    self._behaviors.append(PickupBlock('block0', self._primitives, allow_shelf=True))
    self._behaviors.append(DropBlock(drop_x=1.5, drop_y=1.0))
    for name in ['block1', 'block2']:
        self._behaviors.append(PickupBlock(name, self._primitives))
        self._behaviors.append(PlaceBlock(self._primitives))
    self._behaviors.append(PickupBlock('block0', self._primitives))
    self._behaviors.append(PlaceBlock(self._primitives))
```

Main failure in this version:

- the run now has the right intuition that `block0` must move first
- but it fails in grasping the shelf block



### v10

not fix 

### v11

not fix


### v12

not fix


### Overall evolution

The failure chain is very consistent:

1. `v00-v06`: many local fixes improve direct pick mechanics, but the shelf remains structurally blocked
2. `v09`: the run finally acknowledges that `block0` must move first
3. `v10-v12`: the rest of the history is spent making debugging clearning the shelf block.

