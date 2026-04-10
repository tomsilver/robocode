"""Observation parsing helpers. All magic numbers live here as named constants."""
import numpy as np

# ─── Observation indices ────────────────────────────────────────────────────
# Robot (9 features)
IDX_ROBOT_X = 0
IDX_ROBOT_Y = 1
IDX_ROBOT_THETA = 2
IDX_ROBOT_BASE_RADIUS = 3
IDX_ROBOT_ARM_JOINT = 4
IDX_ROBOT_ARM_LENGTH = 5
IDX_ROBOT_VACUUM = 6
IDX_ROBOT_GRIPPER_HEIGHT = 7
IDX_ROBOT_GRIPPER_WIDTH = 8

# Target surface (10 features, starting at 9)
IDX_SURF_X = 9
IDX_SURF_Y = 10
IDX_SURF_THETA = 11
IDX_SURF_STATIC = 12
IDX_SURF_WIDTH = 17
IDX_SURF_HEIGHT = 18

# Target block (10 features, starting at 19)
IDX_BLOCK_X = 19
IDX_BLOCK_Y = 20
IDX_BLOCK_THETA = 21
IDX_BLOCK_STATIC = 22
IDX_BLOCK_WIDTH = 27
IDX_BLOCK_HEIGHT = 28

# Obstructions: 4 × 10 features, starting at 29
IDX_OBS_BASES = [29, 39, 49, 59]
OBS_X_OFFSET = 0
OBS_Y_OFFSET = 1
OBS_THETA_OFFSET = 2
OBS_WIDTH_OFFSET = 8
OBS_HEIGHT_OFFSET = 9
NUM_OBSTRUCTIONS = 4

# ─── World / physics constants ───────────────────────────────────────────────
WORLD_MIN_X = 0.0
WORLD_MAX_X = 1.618   # golden ratio
TABLE_TOP_Y = 0.1     # y of table surface (= surface_y + surface_height)

# Gripper geometry (from obstruction2d config)
GRIPPER_WIDTH_ALONG_ARM = 0.01   # thin along arm
SUCTION_WIDTH = 0.01             # same as gripper_width
# Suction center offset from robot base along arm direction:
SUCTION_OFFSET = GRIPPER_WIDTH_ALONG_ARM + SUCTION_WIDTH / 2  # = 0.015

# Grasp / place tolerances
VACUUM_THRESHOLD = 0.5
IS_ON_TOL = 0.025      # same as the env's tol in is_on()
GRASP_DIST_TOL = 0.12  # block near gripper when carried (generous)

# Drop zone sizing
DROP_MARGIN = 0.10     # min clearance between drop zone and surface/world edge
DROP_SPACING = 0.20    # x spacing between consecutive drop zones (needs robot_radius clearance)


# ─── Extraction helpers ──────────────────────────────────────────────────────
def extract_robot(obs):
    return {
        'x':            float(obs[IDX_ROBOT_X]),
        'y':            float(obs[IDX_ROBOT_Y]),
        'theta':        float(obs[IDX_ROBOT_THETA]),
        'base_radius':  float(obs[IDX_ROBOT_BASE_RADIUS]),
        'arm_joint':    float(obs[IDX_ROBOT_ARM_JOINT]),
        'arm_length':   float(obs[IDX_ROBOT_ARM_LENGTH]),
        'vacuum':       float(obs[IDX_ROBOT_VACUUM]),
        'gripper_height': float(obs[IDX_ROBOT_GRIPPER_HEIGHT]),
        'gripper_width':  float(obs[IDX_ROBOT_GRIPPER_WIDTH]),
    }


def extract_target_surface(obs):
    return {
        'x':      float(obs[IDX_SURF_X]),
        'y':      float(obs[IDX_SURF_Y]),
        'width':  float(obs[IDX_SURF_WIDTH]),
        'height': float(obs[IDX_SURF_HEIGHT]),
        'theta':  float(obs[IDX_SURF_THETA]),
    }


def extract_target_block(obs):
    return {
        'x':      float(obs[IDX_BLOCK_X]),
        'y':      float(obs[IDX_BLOCK_Y]),
        'width':  float(obs[IDX_BLOCK_WIDTH]),
        'height': float(obs[IDX_BLOCK_HEIGHT]),
        'theta':  float(obs[IDX_BLOCK_THETA]),
    }


def extract_obstruction(obs, i):
    base = IDX_OBS_BASES[i]
    return {
        'x':      float(obs[base + OBS_X_OFFSET]),
        'y':      float(obs[base + OBS_Y_OFFSET]),
        'width':  float(obs[base + OBS_WIDTH_OFFSET]),
        'height': float(obs[base + OBS_HEIGHT_OFFSET]),
        'theta':  float(obs[base + OBS_THETA_OFFSET]),
    }


# ─── Geometric helpers ───────────────────────────────────────────────────────
def rect_center_x(rect):
    """Horizontal center of a rect (valid for theta=0)."""
    return rect['x'] + rect['width'] / 2.0


def rect_center_y(rect):
    """Vertical center of a rect (valid for theta=0)."""
    return rect['y'] + rect['height'] / 2.0


def rect_top(rect):
    """Top y of a rect (valid for theta=0)."""
    return rect['y'] + rect['height']


# Tolerance for determining "obstruction is at table/surface level"
OBS_ON_SURFACE_Y_TOL = 0.05  # obs y within this of surface top → at table level


def x_ranges_overlap(r1, r2):
    """True if r1's x-range overlaps r2's x-range."""
    return r1['x'] < r2['x'] + r2['width'] and r1['x'] + r1['width'] > r2['x']


def rects_overlap_aabb(r1, r2):
    """AABB overlap (ignores rotation — fine for theta≈0 objects)."""
    return (x_ranges_overlap(r1, r2) and
            r1['y'] < r2['y'] + r2['height'] and
            r1['y'] + r1['height'] > r2['y'])


def obstruction_on_surface(obs, i):
    """True if obstruction i is at table level and its x-range overlaps surface.

    Obstructions sit ON the surface (y = surface_top + 1e-6) so a strict AABB
    check fails.  We detect them by checking (a) x-range overlaps and (b) the
    obstruction bottom is within OBS_ON_SURFACE_Y_TOL of the surface top.
    """
    obs_rect  = extract_obstruction(obs, i)
    surf_rect = extract_target_surface(obs)
    surf_top  = surf_rect['y'] + surf_rect['height']
    # Y proximity: obstruction bottom must be near surface top
    if abs(obs_rect['y'] - surf_top) > OBS_ON_SURFACE_Y_TOL:
        return False
    # X overlap
    return x_ranges_overlap(obs_rect, surf_rect)


def any_obstruction_on_surface(obs):
    """True if any obstruction overlaps target surface."""
    return any(obstruction_on_surface(obs, i) for i in range(NUM_OBSTRUCTIONS))


# ─── Grasp / placement predicates ────────────────────────────────────────────
def gripper_center(obs):
    r = extract_robot(obs)
    gx = r['x'] + np.cos(r['theta']) * r['arm_joint']
    gy = r['y'] + np.sin(r['theta']) * r['arm_joint']
    return gx, gy


def is_block_grasped(obs):
    """Vacuum on AND block center is within GRASP_DIST_TOL of gripper center."""
    r = extract_robot(obs)
    if r['vacuum'] < VACUUM_THRESHOLD:
        return False
    gx, gy = gripper_center(obs)
    blk = extract_target_block(obs)
    bx = blk['x'] + blk['width'] / 2.0
    by = blk['y'] + blk['height'] / 2.0
    return np.hypot(bx - gx, by - gy) < GRASP_DIST_TOL


def block_is_on_surface(obs):
    """Replicate the env's is_on() check for theta=0 block/surface."""
    blk = extract_target_block(obs)
    surf = extract_target_surface(obs)
    bx = blk['x']
    by = blk['y']
    bw = blk['width']
    sx = surf['x']
    sy = surf['y']
    sw = surf['width']
    sh = surf['height']
    for vx, vy in [(bx, by), (bx + bw, by)]:
        offset_y = vy - IS_ON_TOL
        if not (sx <= vx <= sx + sw and sy <= offset_y <= sy + sh):
            return False
    return True


# Half-width of carried object: max obstruction half-width ≈ 0.10
CARRY_OBJ_HALF_W = 0.10

# ─── Drop zone computation ────────────────────────────────────────────────────
def get_drop_zones(obs):
    """Return N x-positions for dropping obstructions.

    Scans left→right for positions where:
    - Enough clearance from target surface and non-surface obstructions
    - Robot can safely descend there with the carried object
    """
    surf = extract_target_surface(obs)
    sx = surf['x']
    sw = surf['width']

    ROBOT_BASE_RADIUS = 0.10

    exclusions = []
    # Target surface: must not accidentally deposit onto it
    exclusions.append((sx - DROP_MARGIN - CARRY_OBJ_HALF_W,
                       sx + sw + DROP_MARGIN + CARRY_OBJ_HALF_W))
    # Target block: exclude robot-body radius on both sides of block center
    # so that deposited obs don't interfere with robot descending to pick block.
    blk = extract_target_block(obs)
    pick_x = blk['x'] + blk['width'] / 2.0
    blk_excl_half = ROBOT_BASE_RADIUS + CARRY_OBJ_HALF_W  # = 0.20
    exclusions.append((pick_x - blk_excl_half, pick_x + blk_excl_half))

    def is_blocked(x):
        for (lo, hi) in exclusions:
            if lo <= x <= hi:
                return True
        return False

    def next_clear(x):
        """Jump x past any blocking interval."""
        changed = True
        while changed:
            changed = False
            for (lo, hi) in exclusions:
                if lo <= x <= hi:
                    x = hi + 0.01
                    changed = True
        return x

    zones = []
    # Start at 0.20: robot base_radius (0.10) + CARRY_OBJ_HALF_W (0.10) from world left edge
    x = max(WORLD_MIN_X + 0.20, WORLD_MIN_X + DROP_MARGIN)
    x = next_clear(x)  # skip any initial blocked region
    while x <= WORLD_MAX_X - DROP_MARGIN and len(zones) < NUM_OBSTRUCTIONS:
        if not is_blocked(x):
            zones.append(x)
            x += DROP_SPACING
            x = next_clear(x)
        else:
            x = next_clear(x)

    # Fallback: use positions near world edges
    if len(zones) < NUM_OBSTRUCTIONS:
        # Try right edge
        for i in range(NUM_OBSTRUCTIONS - len(zones)):
            candidate = WORLD_MAX_X - DROP_MARGIN - i * DROP_SPACING
            if not is_blocked(candidate):
                zones.append(candidate)
    # Final fallback: stack at first valid zone (reactive descent handles stacking)
    while len(zones) < NUM_OBSTRUCTIONS:
        zones.append(zones[0] if zones else WORLD_MIN_X + 0.20)

    return zones[:NUM_OBSTRUCTIONS]


def adjust_zone_for_obs(zone_x, carried_half_w, obs_rect):
    """Shift zone_x so carried object [zone_x±half_w] doesn't overlap obs_rect.

    Returns (new_zone_x, adjusted) where adjusted is True if a shift happened.
    Tries moving left first, then right.
    """
    obj_left  = zone_x - carried_half_w
    obj_right = zone_x + carried_half_w
    o_left  = obs_rect['x']
    o_right = obs_rect['x'] + obs_rect['width']
    if obj_right <= o_left or obj_left >= o_right:
        return zone_x, False  # no overlap

    # Try left: place object right edge at o_left - 0.01
    cand_left = o_left - carried_half_w - 0.01
    if cand_left >= WORLD_MIN_X + 0.20:
        return cand_left, True
    # Try right: place object left edge at o_right + 0.01
    cand_right = o_right + carried_half_w + 0.01
    if cand_right <= WORLD_MAX_X - DROP_MARGIN:
        return cand_right, True
    # No good adjustment found, return original
    return zone_x, False


def refine_zones_for_assignments(obs, on_surf_indices, zones):
    """Adjust each zone so the carried object doesn't overlap any non-surface obs."""
    non_surf_rects = []
    for j in range(NUM_OBSTRUCTIONS):
        if j not in on_surf_indices and not obstruction_on_surface(obs, j):
            non_surf_rects.append(extract_obstruction(obs, j))

    refined = list(zones)
    for k, i in enumerate(on_surf_indices):
        obj = extract_obstruction(obs, i)
        half_w = obj['width'] / 2.0
        zone_x = refined[k]
        for ns in non_surf_rects:
            zone_x, _ = adjust_zone_for_obs(zone_x, half_w, ns)
        refined[k] = zone_x
    return refined
