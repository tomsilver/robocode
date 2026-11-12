"""Class-based control policies for milestone 8 and the KinDER policy itself.

Controls run in the disc world. Each plants one claim the analyzer must recover:
planning once at initialization, replanning every call, an unused plan, no search,
exhaustive reachability, and renamed nested storage. Labels come from construction and
are never inputs to discovery or verification.
"""

from .policy_domains import POLICIES

_PLANNER = '''
    def _plan(self, start, goal, obstacles):
        """A* on a 0.1 grid; returns waypoints ending at the exact goal."""
        step = 0.1

        def cell(p):
            return (int(round(p[0] / step)), int(round(p[1] / step)))

        def point(c):
            return (step * c[0], step * c[1])

        def free(c):
            x, y = point(c)
            if not (0.05 <= x <= 2.45 and 0.05 <= y <= 2.45):
                return False
            return all(math.hypot(x - ox, y - oy) > r + 0.1 for ox, oy, r in obstacles)

        root, target = cell(start), cell(goal)
        frontier = [(0.0, root)]
        cost, parent = {root: 0.0}, {root: None}
        while frontier:
            _, cur = heapq.heappop(frontier)
            if cur == target:
                break
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cur[0] + di, cur[1] + dj)
                if not free(nxt):
                    continue
                nc = cost[cur] + math.hypot(di, dj)
                if nc < cost.get(nxt, 1e30):
                    cost[nxt], parent[nxt] = nc, cur
                    h = math.hypot(nxt[0] - target[0], nxt[1] - target[1])
                    heapq.heappush(frontier, (nc + h, nxt))
        path = []
        node = target if target in parent else None
        while node is not None:
            path.append(point(node))
            node = parent[node]
        path.reverse()
        return path + [(goal[0], goal[1])]
'''

_HEADER = """
import heapq
import math


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.path = []
        self.goal = None
"""

_FOLLOW = """
    def _steer(self, x, y, target):
        return (
            max(-0.035, min(0.035, target[0] - x)),
            max(-0.035, min(0.035, target[1] - y)),
        )

    def _follow(self, state):
        x, y = state["robot"]
        while self.path and math.hypot(self.path[0][0] - x, self.path[0][1] - y) < 0.02:
            self.path.pop(0)
        return self._steer(x, y, self.path[0] if self.path else self.goal)
"""

PLAN_ONCE = _HEADER + _PLANNER + _FOLLOW + """
    def reset(self, state, info):
        self.goal = tuple(state["goal"])
        self.path = self._plan(state["robot"], self.goal, state["obstacles"])

    def get_action(self, state):
        return self._follow(state)
"""

# The old shape-link + rotation checks falsely credited this search. The cache
# really controls actions, but it is built independently after discarding search.
DISCARDED_SEARCH = PLAN_ONCE.replace(
    '        self.path = self._plan(state["robot"], self.goal, state["obstacles"])',
    '        self.path = self._plan(state["robot"], self.goal, state["obstacles"])\n'
    "        self.path = [(0.1 * round(self.goal[0] / 0.1), "
    "0.1 * round(self.goal[1] / 0.1)), self.goal]",
)

# Stores the path like the KinDER policy: the endpoint cell is dropped and the
# exact goal (which lies inside that cell) is appended instead.
ENDPOINT_REPLACED = PLAN_ONCE.replace(
    "        return path + [(goal[0], goal[1])]",
    "        return path[:-1] + [(goal[0], goal[1])]",
)

# A* whose heap entries carry a tie-break counter: (f, push number, cell).
TIEBREAK_ASTAR = (
    PLAN_ONCE.replace("frontier = [(0.0, root)]", "frontier = [(0.0, 0, root)]")
    .replace("_, cur = heapq.heappop(frontier)", "_, _, cur = heapq.heappop(frontier)")
    .replace(
        "heapq.heappush(frontier, (nc + h, nxt))",
        "heapq.heappush(frontier, (nc + h, len(cost), nxt))",
    )
)

_BFS = """
        root, target = cell(start), cell(goal)
        frontier = deque([root])
        cost, parent = {root: 0.0}, {root: None}
        while frontier:
            cur = frontier.popleft()
            if cur == target:
                break
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cur[0] + di, cur[1] + dj)
                if not free(nxt) or nxt in parent:
                    continue
                cost[nxt], parent[nxt] = cost[cur] + 1.0, cur
                frontier.append(nxt)
"""

_ASTAR = PLAN_ONCE[
    PLAN_ONCE.index("        root, target = cell(start), cell(goal)") : PLAN_ONCE.index(
        "        path = []"
    )
]

# Breadth-first search with a deque frontier (and a cost map).
DEQUE_BFS = (
    PLAN_ONCE.replace(_ASTAR, _BFS)
    .replace("import heapq\n", "import heapq\nfrom collections import deque\n")
    .replace("A* on a 0.1 grid;", "Breadth-first search on a 0.1 grid;")
)

# Breadth-first search that keeps no cost map: outside the grammar. Its worklist
# must make it unreadable (unknown), never zero planning.
BFS_WITHOUT_COSTS = DEQUE_BFS.replace(
    "        cost, parent = {root: 0.0}, {root: None}", "        parent = {root: None}"
).replace(
    "                cost[nxt], parent[nxt] = cost[cur] + 1.0, cur",
    "                parent[nxt] = cur",
)

REPLAN_EVERY_CALL = _HEADER + _PLANNER + _FOLLOW + """
    def reset(self, state, info):
        self.goal = tuple(state["goal"])

    def get_action(self, state):
        # The first waypoint is the robot's own cell; head for the next one.
        self.path = self._plan(state["robot"], self.goal, state["obstacles"])[1:]
        return self._follow(state)
"""

UNUSED_PLAN = _HEADER + _PLANNER + _FOLLOW + """
    def reset(self, state, info):
        self.goal = tuple(state["goal"])
        self.path = self._plan(state["robot"], self.goal, state["obstacles"])

    def get_action(self, state):
        x, y = state["robot"]
        return self._steer(x, y, self.goal)
"""

REACTIVE = """
import math


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.goal = None

    def reset(self, state, info):
        self.goal = tuple(state["goal"])

    def get_action(self, state):
        x, y = state["robot"]
        dx, dy = self.goal[0] - x, self.goal[1] - y
        for ox, oy, r in state["obstacles"]:
            if math.hypot(x - ox, y - oy) < r + 0.15:
                dx, dy = dx - (oy - y), dy + (ox - x)
        return (max(-0.035, min(0.035, dx)), max(-0.035, min(0.035, dy)))
"""

REACH_THEN_PICK = _HEADER + _FOLLOW + """
    def reset(self, state, info):
        self.goal = tuple(state["goal"])
        step = 0.1
        obstacles = state["obstacles"]

        def cell(p):
            return (int(round(p[0] / step)), int(round(p[1] / step)))

        def point(c):
            return (step * c[0], step * c[1])

        def free(c):
            x, y = point(c)
            if not (0.05 <= x <= 2.45 and 0.05 <= y <= 2.45):
                return False
            return all(math.hypot(x - ox, y - oy) > r + 0.1 for ox, oy, r in obstacles)

        start = cell(state["robot"])
        frontier = [(0.0, start)]
        cost, parent = {start: 0.0}, {start: None}
        while frontier:
            _, cur = heapq.heappop(frontier)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cur[0] + di, cur[1] + dj)
                if not free(nxt):
                    continue
                nc = cost[cur] + math.hypot(di, dj)
                if nc < cost.get(nxt, 1e30):
                    cost[nxt], parent[nxt] = nc, cur
                    heapq.heappush(frontier, (nc, nxt))
        best = min(
            cost,
            key=lambda c: math.hypot(point(c)[0] - self.goal[0], point(c)[1] - self.goal[1]),
        )
        cells = []
        while best is not None:
            cells.append(best)
            best = parent[best]
        self.path = [point(c) for c in reversed(cells)] + [self.goal]

    def get_action(self, state):
        return self._follow(state)
"""

NESTED_RENAMED = _HEADER.replace("self.path = []", "self.route = []") + """
    def _make(self, here, there, discs):
        size = 0.1

        def snap(p):
            return (int(round(p[0] / size)), int(round(p[1] / size)))

        def spot(c):
            return (size * c[0], size * c[1])

        def ok(c):
            x, y = spot(c)
            if not (0.05 <= x <= 2.45 and 0.05 <= y <= 2.45):
                return False
            return all(math.hypot(x - ox, y - oy) > r + 0.1 for ox, oy, r in discs)

        a, z = snap(here), snap(there)
        box = {"q": [(0.0, a)], "g": {a: 0.0}, "p": {a: None}}
        while box["q"]:
            _, u = heapq.heappop(box["q"])
            if u == z:
                break
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v = (u[0] + di, u[1] + dj)
                if not ok(v):
                    continue
                w = box["g"][u] + math.hypot(di, dj)
                if w < box["g"].get(v, 1e30):
                    box["g"][v], box["p"][v] = w, u
                    k = math.hypot(v[0] - z[0], v[1] - z[1])
                    heapq.heappush(box["q"], (w + k, v))
        out = []
        n = z if z in box["p"] else None
        while n is not None:
            out.append(spot(n))
            n = box["p"][n]
        out.reverse()
        return out + [(there[0], there[1])]

    def reset(self, state, info):
        self.goal = tuple(state["goal"])
        self.route = self._make(state["robot"], self.goal, state["obstacles"])

    def get_action(self, state):
        x, y = state["robot"]
        while self.route and math.hypot(self.route[0][0] - x, self.route[0][1] - y) < 0.02:
            self.route.pop(0)
        t = self.route[0] if self.route else self.goal
        return (max(-0.035, min(0.035, t[0] - x)), max(-0.035, min(0.035, t[1] - y)))
"""

# A* written the way GPT-6 Astra's KinDER policy writes it: a numpy obstacle grid
# (`bad`), a parent table without a root entry, (f, g, cell) heap entries, a weighted
# octile heuristic with 1.414 diagonals, no corner cutting, a return inside the loop.
_GRID_OCTILE = """
    def _plan(self, start, goal, obstacles):
        \"\"\"Weighted-octile A* over a numpy occupancy grid.\"\"\"
        step, n = 0.1, 26
        axis = np.arange(n) * step
        cx, cy = np.meshgrid(axis, axis, indexing="ij")
        bad = (cx < 0.05) | (cx > 2.45) | (cy < 0.05) | (cy > 2.45)
        for ox, oy, r in obstacles:
            bad |= np.hypot(cx - ox, cy - oy) <= r + 0.1
        src = tuple(np.clip(np.round(np.asarray(start) / step).astype(int), 0, n - 1))
        dst = tuple(np.clip(np.round(np.asarray(goal) / step).astype(int), 0, n - 1))
        if bad[dst]:
            return [(goal[0], goal[1])]
        front = [(0.0, 0.0, src)]
        cost = {src: 0.0}
        parent = {}
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        while front:
            _, g, k = heapq.heappop(front)
            if g > cost[k] + 1e-6:
                continue
            if k == dst:
                out = [(goal[0], goal[1])]
                while k != src:
                    out.append((step * k[0], step * k[1]))
                    k = parent[k]
                out.reverse()
                return out
            for dx, dy, d in moves:
                nxt = (k[0] + dx, k[1] + dy)
                if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or bad[nxt]:
                    continue
                if dx and dy and (bad[k[0] + dx, k[1]] or bad[k[0], k[1] + dy]):
                    continue
                ng = g + d
                if ng >= cost.get(nxt, 1e20):
                    continue
                cost[nxt] = ng
                parent[nxt] = k
                ax, ay = abs(dst[0] - nxt[0]), abs(dst[1] - nxt[1])
                h = max(ax, ay) + .414 * min(ax, ay)
                heapq.heappush(front, (ng + h * 1.02, ng, nxt))
        return [(goal[0], goal[1])]
"""

GRID_OCTILE = (
    _HEADER.replace("import math\n", "import math\n\nimport numpy as np\n")
    + _GRID_OCTILE
    + _FOLLOW
    + """
    def reset(self, state, info):
        self.goal = tuple(state["goal"])
        self.path = self._plan(state["robot"], self.goal, state["obstacles"])

    def get_action(self, state):
        return self._follow(state)
"""
)

POLICY_FIXTURES = {
    "plan_once": PLAN_ONCE,
    "discarded_search": DISCARDED_SEARCH,
    "replan_every_call": REPLAN_EVERY_CALL,
    "unused_plan": UNUSED_PLAN,
    "reactive": REACTIVE,
    "reach_then_pick": REACH_THEN_PICK,
    "nested_renamed": NESTED_RENAMED,
    "endpoint_replaced": ENDPOINT_REPLACED,
    "tiebreak_astar": TIEBREAK_ASTAR,
    "deque_bfs": DEQUE_BFS,
    "bfs_without_costs": BFS_WITHOUT_COSTS,
    "grid_octile": GRID_OCTILE,
}

GROUND_TRUTH = {
    "plan_once": "A* once at initialization; follows the cached path",
    "discarded_search": (
        "Discards A* result; follows independently constructed goal waypoints"
    ),
    "replan_every_call": "A* in every action call",
    "unused_plan": "A* at initialization; steers straight to the goal instead",
    "reactive": "Steers toward the goal; no search",
    "reach_then_pick": "Explores every reachable cell, then follows a tree path",
    "nested_renamed": "plan_once with renamed, nested search storage",
    "endpoint_replaced": (
        "plan_once storing its path without the goal cell, then the exact goal"
    ),
    "tiebreak_astar": "plan_once with (f, tie-break counter, cell) heap entries",
    "deque_bfs": "Breadth-first search with a deque frontier; plan once and follow",
    "bfs_without_costs": (
        "Breadth-first search without a cost map: outside the grammar, so unknown"
    ),
    "grid_octile": (
        "A* in Astra's style: numpy obstacle grid, empty parent table, weighted "
        "octile heuristic, 1.414 diagonals, no corner cutting; plan once and follow"
    ),
    "clutteredretrieval2d": "Agent-written KinDER policy; no construction label",
}

KINDER_POLICY = "clutteredretrieval2d"


def kinder_source() -> str:
    """The byte-identical agent-written policy (see policies/PROVENANCE.md)."""
    return (POLICIES / "clutteredretrieval2d_codex_gpt56sol.py.txt").read_text(
        encoding="utf-8"
    )
