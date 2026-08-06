"""Approach for StickButton2DEnv (variable number of buttons).

Key facts about the environment
-------------------------------
* World is 3.5 x 2.5.  A static "table" covers y >= 1.25 and blocks only the
  robot *base* (ZOrder.ALL).  The arm/gripper (SURFACE) and the stick
  (SURFACE) pass over it freely.
* Buttons have ZOrder.NONE -> they collide with nothing.  A button is
  "pressed" as soon as its circle geometrically overlaps any robot body
  (base / arm / gripper) or any part of the stick.
* The stick is a 0.05 x 1.25 movable rectangle, initially axis aligned
  (vertical).  Grasping it at its bottom edge with the robot pointing +y
  turns the robot into a ~1.5 long vertical bar that can reach the whole
  arena in y.
* A colliding motion is rejected in full (state unchanged).

Strategy
--------
If every button can be touched by the robot base alone (all of them are low
enough), we skip the stick entirely and do a base-only tour, routing around
the stick which is then an obstacle.  Otherwise we grasp the stick and sweep
the resulting vertical bar over the buttons in a good order.
"""

import time

import numpy as np

PI = np.pi

WORLD_X_MIN, WORLD_X_MAX = 0.0, 3.5
WORLD_Y_MIN, WORLD_Y_MAX = 0.0, 2.5
TABLE_Y = 1.25


def _wrap(a):
    return (a + PI) % (2 * PI) - PI


class GeneratedApproach:

    DETOUR_TOL = 0.30

    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
        self.lo = np.asarray(action_space.low, dtype=np.float64)
        self.hi = np.asarray(action_space.high, dtype=np.float64)

    # ------------------------------------------------------------------ utils
    def _parse(self, state):
        robot = None
        stick = None
        rects = []
        buttons = []
        for name in state.get_object_names():
            obj = state.get_object_from_name(name)
            tname = obj.type.name
            if tname == "crv_robot":
                robot = obj
            elif tname == "circle":
                buttons.append((name, obj))
            elif tname in ("rectangle", "double_rectangle", "lobject"):
                rects.append((name, obj))
        # The stick is the movable rectangle (prefer the one named "stick").
        for name, obj in rects:
            if name == "stick":
                stick = obj
                break
        if stick is None:
            for name, obj in rects:
                try:
                    movable = float(state.get(obj, "static")) < 0.5
                except Exception:
                    movable = True
                if movable:
                    stick = obj
                    break
        if stick is None and rects:
            stick = rects[0][1]
        named = [kv for kv in buttons if kv[0].startswith("button")]
        if named:
            buttons = named
        buttons.sort(key=lambda kv: kv[0])
        return robot, stick, [b for _, b in buttons]

    def _pressed(self, state, b):
        return float(state.get(b, "color_g")) > 0.45

    def _act(self, dx=0.0, dy=0.0, dth=0.0, darm=0.0, vac=0.0):
        a = np.array([dx, dy, dth, darm, vac], dtype=np.float32)
        return np.clip(a, self.lo, self.hi).astype(np.float32)

    # ------------------------------------------------------------------ reset
    def reset(self, state, info):
        self._t0 = time.time()
        robot, stick, buttons = self._parse(state)
        self.br = float(state.get(robot, "base_radius"))
        self.gw = float(state.get(robot, "gripper_width"))
        self.gh = float(state.get(robot, "gripper_height"))
        self.arm_max = float(state.get(robot, "arm_length"))
        self.W = float(state.get(stick, "width"))
        self.H = float(state.get(stick, "height"))
        self.R = float(state.get(buttons[0], "radius")) if buttons else 0.05

        # Conservative radius of the whole robot with the arm retracted.
        self.rr = float(
            np.hypot(self.br + self.gw / 2.0, self.gh / 2.0)
        ) + 0.004

        self.x_min = WORLD_X_MIN + self.br + 0.006
        self.x_max = WORLD_X_MAX - self.br - 0.006
        self.y_min = WORLD_Y_MIN + self.br + 0.006
        self.y_max_base = TABLE_Y - self.br - 0.008

        self.sx0 = float(state.get(stick, "x"))
        self.sy0 = float(state.get(stick, "y"))
        # Grasp pose: the suction band [ry+br+gw, ry+br+2gw] straddles the
        # bottom edge of the stick, gripper stays clear of it.
        self.grasp_y = self.sy0 - self.br - 1.5 * self.gw
        self.grasp_y = min(max(self.grasp_y, self.y_min), self.y_max_base)
        self.grasp_x = min(max(self.sx0 + self.W / 2.0, self.x_min), self.x_max)

        # Stick-as-obstacle description (used only in base-only mode).
        self.band_lo = self.sx0 - self.rr
        self.band_hi = self.sx0 + self.W + self.rr
        self.corridor_y = self.sy0 - self.rr - 0.004

        self.dx_off = None
        self.dy_off = None
        self.ry_max = self.y_max_base
        self.stuck = 0
        self.last_pose = None
        self.arrive_ticks = 0
        self.sweep_dir = 1
        self.t = 0
        self._next_replan = 700
        self.regrasps = 0
        self.grasp_ticks = 0
        self.wps = []
        self.cur_target = None
        self.order = None
        self.arm_target = 0.0

        # ---- decide the mode -------------------------------------------
        self.avoid_stick = True
        self.planning_base = True
        rx0 = float(state.get(robot, "x"))
        ry0 = float(state.get(robot, "y"))
        self.mode = "base"
        base_ok = []
        for b in buttons:
            if self._pressed(state, b):
                continue
            if self._base_pos_for(state, b) is None:
                self.mode = "stick"
            else:
                base_ok.append(b)
        if self.mode == "base":
            self.phase = "tour"
            self.order = self._plan_order(state, buttons, rx0, ry0)
            self.pre_order = []
        else:
            # Predicted stick offsets once grasped (used for planning).
            self.dx_off = self.sx0 - self.grasp_x
            self.dy_off = self.sy0 - self.grasp_y
            self.ry_max = max(self.y_min, min(
                self.y_max_base,
                WORLD_Y_MAX - self.dy_off - self.H - 0.012))
            unpressed = [b for b in buttons if not self._pressed(state, b)]
            goal = (self.grasp_x, self.grasp_y)

            def total(pre):
                self.planning_base = True
                self.avoid_stick = True
                cur = (rx0, ry0)
                c = 0.0
                for b in pre:
                    p = self._base_pos_for(state, b, cur[0], cur[1])
                    if p is None:
                        return None, None
                    p = (p[0], p[1])
                    c += self._travel_cost(cur, p)
                    cur = p
                c += self._travel_cost(cur, goal)
                self.planning_base = False
                self.avoid_stick = False
                rest = [b for b in unpressed if b not in pre]
                order = self._plan_order(state, rest, goal[0], goal[1],
                                         refine=len(unpressed) <= 9)
                c += self._tour_cost(state, order, goal)
                return c, order

            self.pre_order = []
            best_c, best_order = total([])
            # Only consider the base-reachable buttons closest to the
            # start -> grasp path, to bound planning time.
            self.planning_base = True
            self.avoid_stick = True
            scored = []
            for b in base_ok:
                p = self._base_pos_for(state, b, rx0, ry0)
                if p is None:
                    continue
                d = (self._travel_cost((rx0, ry0), (p[0], p[1]))
                     + self._travel_cost((p[0], p[1]), goal))
                scored.append((d, id(b), b))
            scored.sort()
            pool = [b for _, _, b in scored[:8]]
            while pool and len(self.pre_order) < 6:
                if time.time() - self._t0 > 3.0:
                    break
                cand_best = None
                for b in pool:
                    c, o = total(self.pre_order + [b])
                    if c is None:
                        continue
                    if cand_best is None or c < cand_best[0]:
                        cand_best = (c, b, o)
                if cand_best is None or cand_best[0] >= best_c - 1e-6:
                    break
                best_c, best_order = cand_best[0], cand_best[2]
                self.pre_order.append(cand_best[1])
                pool.remove(cand_best[1])
            self.planning_base = True
            self.avoid_stick = True
            self.phase = "pretour" if self.pre_order else "approach"

    # ------------------------------------------------- base-only mode helpers
    def _blocked(self, px, py):
        if not self.avoid_stick:
            return False
        return (self.band_lo < px < self.band_hi) and py > self.corridor_y

    def _arm_blocked(self, px, top_y):
        if not self.avoid_stick:
            return False
        half = self.gh / 2.0 + 0.012
        return (self.sx0 - half < px < self.sx0 + self.W + half
                and top_y > self.sy0)

    def _base_pos_for(self, state, b, cx=None, cy=None):
        """Best legal, stick-free base pose (x, y, arm_joint) touching b."""
        bx = float(state.get(b, "x"))
        by = float(state.get(b, "y"))
        reach = self.br + self.R - 0.012
        best = None
        cands = [(bx, by)]
        for rad in (0.05, 0.09, 0.13):
            for k in range(16):
                a = 2 * PI * k / 16.0
                cands.append((bx + rad * np.cos(a), by + rad * np.sin(a)))
        for px, py in cands:
            px = min(max(px, self.x_min), self.x_max)
            py = min(max(py, self.y_min), self.y_max_base)
            if np.hypot(px - bx, py - by) > reach:
                continue
            if self._blocked(px, py):
                continue
            if cx is None:
                return (px, py, self.br)
            c = self._travel_cost((cx, cy), (px, py))
            if best is None or c < best[0]:
                best = (c, (px, py, self.br))
        if best is not None:
            return best[1]
        # Fall back to reaching with the arm pointed straight up (+y).
        if abs(bx - min(max(bx, self.x_min), self.x_max)) > \
                self.gh / 2.0 + self.R - 0.014:
            return None
        px = min(max(bx, self.x_min), self.x_max)
        tolg = self.gw / 2.0 + self.R - 0.014
        lo = by - self.arm_max - tolg
        hi = by - self.br + tolg
        lo = max(lo, self.y_min)
        hi = min(hi, self.y_max_base)
        if hi < lo:
            return None
        py = min(max(cy if cy is not None else lo, lo), hi)
        aj = min(max(by - py, self.br), self.arm_max)
        if abs(by - (py + aj)) > tolg:
            return None
        if self._blocked(px, py) or self._arm_blocked(px, py + aj + self.gw):
            return None
        return (px, py, aj)

    def _x_crosses_band(self, x1, x2):
        return min(x1, x2) < self.band_hi and max(x1, x2) > self.band_lo

    def _route(self, a, b):
        if not self.avoid_stick:
            return [b]
        ax, ay = a
        bx, by = b
        if not self._x_crosses_band(ax, bx):
            return [b]
        cy = self.corridor_y
        wps = []
        if ay > cy:
            wps.append((ax, cy))
        if by > cy:
            wps.append((bx, cy))
        wps.append((bx, by))
        return wps

    def _travel_cost(self, a, b):
        wps = self._route(a, b)
        c = 0.0
        cur = a
        for w in wps:
            c += max(abs(w[0] - cur[0]), abs(w[1] - cur[1]))
            cur = w
        return c

    # -------------------------------------------------- stick mode geometry
    def _target_rx(self, bx):
        want = bx - (self.dx_off + self.W / 2.0)
        return min(max(want, self.x_min), self.x_max)

    def _feasible_ry(self, bx, by, rx):
        out = []
        sxl = rx + self.dx_off
        sxr = sxl + self.W
        if bx + self.R > sxl + 0.004 and bx - self.R < sxr - 0.004:
            lo = by - self.R - self.dy_off - self.H + 0.012
            hi = by + self.R - self.dy_off - 0.012
            if hi > lo:
                out.append((lo, hi))
        d = self.br + self.R - 0.012
        ddx = abs(bx - rx)
        if ddx < d:
            h = float(np.sqrt(max(d * d - ddx * ddx, 0.0)))
            out.append((by - h, by + h))
        res = []
        for lo, hi in out:
            lo2 = max(lo, self.y_min)
            hi2 = min(hi, self.ry_max)
            if hi2 >= lo2:
                res.append((lo2, hi2))
        return res

    @staticmethod
    def _pick(ivs, pref, lo_lim=None, hi_lim=None):
        best = None
        for lo, hi in ivs:
            if lo_lim is not None:
                lo = max(lo, lo_lim)
                hi = min(hi, hi_lim)
                if hi < lo:
                    continue
            v = min(max(pref, lo), hi)
            c = abs(v - pref)
            if best is None or c < best[0]:
                best = (c, v)
        return None if best is None else best[1]

    def _target_ry(self, bx, by, rx, cur_ry):
        ivs = self._feasible_ry(bx, by, rx)
        if not ivs:
            return min(max(by - 0.7, self.y_min), self.ry_max)
        best = None
        for lo, hi in ivs:
            v = min(max(cur_ry, lo), hi)
            c = abs(v - cur_ry)
            if best is None or c < best[0]:
                best = (c, v)
        return best[1]

    # ------------------------------------------------------------- targeting
    def _target_pose(self, state, b, cx, cy):
        """Robot (x, y) that presses button b, given we are at (cx, cy)."""
        bx = float(state.get(b, "x"))
        by = float(state.get(b, "y"))
        if self.planning_base:
            p = self._base_pos_for(state, b, cx, cy)
            if p is None:
                return (min(max(bx, self.x_min), self.x_max),
                        min(max(by, self.y_min), self.y_max_base))
            return (p[0], p[1])
        tx = self._target_rx(bx)
        ty = self._target_ry(bx, by, tx, cy)
        return (tx, ty)

    def _tour_cost(self, state, order, start):
        cur = start
        tot = 0.0
        for b in order:
            p = self._target_pose(state, b, cur[0], cur[1])
            tot += self._travel_cost(cur, p)
            cur = p
        return tot

    def _plan_order(self, state, buttons, rx, ry, refine=True):
        remaining = [b for b in buttons if not self._pressed(state, b)]
        order = []
        cur = (rx, ry)
        while remaining:
            best = None
            for b in remaining:
                p = self._target_pose(state, b, cur[0], cur[1])
                c = self._travel_cost(cur, p)
                if best is None or c < best[0]:
                    best = (c, b, p)
            order.append(best[1])
            remaining.remove(best[1])
            cur = best[2]
        # 2-opt / or-opt refinement (cheap; n is small).
        if refine and 2 < len(order) <= 30:
            start = (rx, ry)
            best_cost = self._tour_cost(state, order, start)
            improved = True
            it = 0
            n = len(order)
            while improved and it < 12 and time.time() - self._t0 < 4.0:
                improved = False
                it += 1
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                        c = self._tour_cost(state, cand, start)
                        if c < best_cost - 1e-6:
                            order = cand
                            best_cost = c
                            improved = True
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        cand = order[:i] + order[i + 1:]
                        cand = cand[:j] + [order[i]] + cand[j:]
                        c = self._tour_cost(state, cand, start)
                        if c < best_cost - 1e-6:
                            order = cand
                            best_cost = c
                            improved = True
        return order

    # ----------------------------------------------------------------- action
    def get_action(self, state):
        self.t += 1
        # Watchdog: if something went badly wrong, replan from scratch.
        if self.t >= self._next_replan:
            self._next_replan = self.t + 400
            keep = self._next_replan
            self.reset(state, {})
            self.t = keep - 400
            self._next_replan = keep
        robot, stick, buttons = self._parse(state)
        rx = float(state.get(robot, "x"))
        ry = float(state.get(robot, "y"))
        th = float(state.get(robot, "theta"))
        aj = float(state.get(robot, "arm_joint"))
        sx = float(state.get(stick, "x"))
        sy = float(state.get(stick, "y"))

        pose = (round(rx, 6), round(ry, 6), round(th, 6), round(aj, 6))
        if self.last_pose is not None and pose == self.last_pose:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_pose = pose

        darm = float(np.clip(self.br - aj, self.lo[3], self.hi[3]))
        dth = float(np.clip(_wrap(PI / 2 - th), self.lo[2], self.hi[2]))

        if self.phase == "pretour":
            tgt = None
            for b in self.pre_order:
                if not self._pressed(state, b):
                    tgt = b
                    break
            if tgt is None:
                self.phase = "approach"
                self.cur_target = None
                self.wps = []
            else:
                if tgt is not self.cur_target:
                    self.cur_target = tgt
                    p = self._base_pos_for(state, tgt, rx, ry)
                    if p is None:
                        p = (self.grasp_x, self.grasp_y, self.br)
                    self.arm_target = p[2]
                    self.wps = self._route((rx, ry), (p[0], p[1]))
                while len(self.wps) > 1 and (
                        abs(rx - self.wps[0][0]) < 3e-3
                        and abs(ry - self.wps[0][1]) < 3e-3):
                    self.wps.pop(0)
                tx, ty = self.wps[0]
                if (abs(rx - tx) < 3e-3 and abs(ry - ty) < 3e-3
                        and len(self.wps) == 1):
                    self.arrive_ticks += 1
                    if self.arrive_ticks > 3:
                        # give up on this one; the stick will get it later
                        self.pre_order = [b for b in self.pre_order
                                          if b is not tgt]
                        self.cur_target = None
                        self.arrive_ticks = 0
                else:
                    self.arrive_ticks = 0
                dx = float(np.clip(tx - rx, self.lo[0], self.hi[0]))
                dy = float(np.clip(ty - ry, self.lo[1], self.hi[1]))
                if (self.arm_target > self.br + 1e-6 and len(self.wps) == 1
                        and abs(rx - tx) < 0.07 and abs(ry - ty) < 0.07
                        and abs(_wrap(PI / 2 - th)) < 0.06):
                    darm = float(np.clip(self.arm_target - aj,
                                         self.lo[3], self.hi[3]))
                if self.stuck >= 2:
                    m = (self.stuck // 2) % 3
                    if m == 0:
                        dy = 0.0
                    elif m == 1:
                        dx = 0.0
                    else:
                        dy = -0.05
                return self._act(dx, dy, dth, darm, 0.0)

        # ------------------------------------------------ stick approach
        if self.phase == "approach":
            if ry > self.grasp_y + 1e-3:
                dy = float(np.clip(self.grasp_y - ry, self.lo[1], self.hi[1]))
                # Make x progress too, but never enter the stick's x band
                # while we are still too high.
                if rx <= self.band_lo:
                    tx = max(rx, min(self.grasp_x, self.band_lo - 0.004))
                elif rx >= self.band_hi:
                    tx = min(rx, max(self.grasp_x, self.band_hi + 0.004))
                else:
                    tx = rx
                dx = float(np.clip(tx - rx, self.lo[0], self.hi[0]))
                if self.stuck >= 3:
                    dx = 0.05 if (self.stuck // 3) % 2 else -0.05
                return self._act(dx, dy, dth, darm, 0.0)
            dx = float(np.clip(self.grasp_x - rx, self.lo[0], self.hi[0]))
            dy = float(np.clip(self.grasp_y - ry, self.lo[1], self.hi[1]))
            aligned = (abs(rx - self.grasp_x) < 2e-3
                       and abs(ry - self.grasp_y) < 2e-3
                       and abs(_wrap(PI / 2 - th)) < 5e-3
                       and abs(aj - self.br) < 1e-3)
            if aligned:
                self.phase = "grasp"
                self.grasp_ticks = 0
            else:
                if self.stuck >= 3:
                    dy = -0.05
                # Vacuum on already: harmless and may grab a step earlier.
                return self._act(dx, dy, dth, darm, 0.0)

        if self.phase == "grasp":
            self.grasp_ticks += 1
            if self.grasp_ticks >= 1:
                self.dx_off = sx - rx
                self.dy_off = sy - ry
                self.ry_max = min(
                    self.y_max_base,
                    WORLD_Y_MAX - self.dy_off - self.H - 0.012,
                )
                self.ry_max = max(self.ry_max, self.y_min)
                self.phase = "tour"
                self.avoid_stick = False
                self.planning_base = False
                self.order = self._plan_order(state, buttons, rx, ry)
                self.cur_target = None
                self.arrive_ticks = 0
            return self._act(0.0, 0.0, dth, darm, 1.0)

        # ---------------------------------------------------------- touring
        vac = 1.0 if self.mode == "stick" else 0.0

        if self.mode == "stick":
            if (abs((sx - rx) - self.dx_off) > 0.02
                    or abs((sy - ry) - self.dy_off) > 0.02):
                if self.regrasps < 3:
                    self.regrasps += 1
                    self.grasp_x = min(max(sx + self.W / 2.0, self.x_min),
                                       self.x_max)
                    gy = sy - self.br - 1.5 * self.gw
                    self.grasp_y = min(max(gy, self.y_min), self.y_max_base)
                    self.phase = "approach"
                    self.avoid_stick = True
                    self.planning_base = True
                    return self._act(0.0, 0.0, dth, darm, 0.0)

        target = None
        for b in self.order:
            if not self._pressed(state, b):
                target = b
                break
        if target is None:
            return self._act(0.0, 0.0, 0.0, 0.0, vac)

        if target is not self.cur_target:
            self.cur_target = target
            self.arrive_ticks = 0
            p = self._target_pose(state, target, rx, ry)
            self.arm_target = self.br
            if self.planning_base:
                q = self._base_pos_for(state, target, rx, ry)
                if q is not None:
                    self.arm_target = q[2]
            self.wps = self._route((rx, ry), (p[0], p[1]))

        bx = float(state.get(target, "x"))
        by = float(state.get(target, "y"))

        # Look-ahead: pick the y that also helps for the *next* button, as
        # long as it costs no extra steps (y motion is free while moving x).
        self.look_ry = None
        if self.mode == "stick" and self.order is not None:
            nxt = None
            seen = False
            for b in self.order:
                if b is target:
                    seen = True
                    continue
                if seen and not self._pressed(state, b):
                    nxt = b
                    break
            if nxt is not None:
                tx0 = self._target_rx(bx)
                ivs = self._feasible_ry(bx, by, tx0)
                ty0 = self._pick(ivs, ry)
                if ty0 is not None:
                    nbx = float(state.get(nxt, "x"))
                    nby = float(state.get(nxt, "y"))
                    ntx = self._target_rx(nbx)
                    nivs = self._feasible_ry(nbx, nby, ntx)
                    p2 = self._pick(nivs, ty0)
                    if p2 is not None:
                        lim = max(abs(tx0 - rx), abs(ty0 - ry))
                        v = self._pick(ivs, p2, ry - lim, ry + lim)
                        if v is not None:
                            self.look_ry = v

        # follow waypoints
        while len(self.wps) > 1 and (
                abs(rx - self.wps[0][0]) < 3e-3
                and abs(ry - self.wps[0][1]) < 3e-3):
            self.wps.pop(0)
        if not self.wps:
            self.wps = [self._target_pose(state, target, rx, ry)]
        tx, ty = self.wps[0]
        if self.look_ry is not None and len(self.wps) == 1:
            ty = self.look_ry

        final = len(self.wps) == 1
        if final and abs(rx - tx) < 3e-3 and abs(ry - ty) < 3e-3:
            self.arrive_ticks += 1
        elif final:
            self.arrive_ticks = 0

        if self.arrive_ticks > 2:
            # Should already have pressed -> sweep in y within the legal band.
            if self.mode == "stick":
                ivs = self._feasible_ry(bx, by, tx)
            else:
                ivs = []
            if ivs:
                lo = min(i[0] for i in ivs)
                hi = max(i[1] for i in ivs)
            else:
                lo, hi = self.y_min, self.ry_max
            if self.sweep_dir > 0 and ry >= hi - 3e-3:
                self.sweep_dir = -1
            elif self.sweep_dir < 0 and ry <= lo + 3e-3:
                self.sweep_dir = 1
            ty = hi if self.sweep_dir > 0 else lo
            if self.arrive_ticks > 40:
                darm = self.hi[3]
                if self.mode == "base":
                    dth = float(np.clip(
                        _wrap(np.arctan2(by - ry, bx - rx) - th),
                        self.lo[2], self.hi[2]))

        dx = float(np.clip(tx - rx, self.lo[0], self.hi[0]))
        dy = float(np.clip(ty - ry, self.lo[1], self.hi[1]))

        if self.mode == "base":
            dth = float(np.clip(_wrap(PI / 2 - th), self.lo[2], self.hi[2]))
            darm = float(np.clip(self.br - aj, self.lo[3], self.hi[3]))
            if (self.arm_target > self.br + 1e-6 and len(self.wps) == 1
                    and abs(rx - tx) < 0.07 and abs(ry - ty) < 0.07
                    and abs(_wrap(PI / 2 - th)) < 0.06):
                darm = float(np.clip(self.arm_target - aj,
                                     self.lo[3], self.hi[3]))
            if self.arrive_ticks > 40:
                dth = float(np.clip(
                    _wrap(np.arctan2(by - ry, bx - rx) - th),
                    self.lo[2], self.hi[2]))
                darm = self.hi[3]

        if self.stuck >= 2:
            m = (self.stuck // 2) % 4
            if m == 0:
                dy = 0.0
            elif m == 1:
                dx = 0.0
            elif m == 2:
                dy = -0.05
            else:
                dx = 0.0
                dy = 0.05

        return self._act(dx, dy, dth, darm, vac)
