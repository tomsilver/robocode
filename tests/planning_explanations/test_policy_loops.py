"""Issue 2: a loop the analyzer cannot read is unknown, never zero planning."""

import ast
import json
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robocode.planning_explanations.search.policy_discovery import (
    LIBRARY,
    UNREADABLE,
    PolicyLimits,
    discover_policy,
    validate_policy,
)
from robocode.planning_explanations.search.policy_domains import DiscWorld
from robocode.planning_explanations.search.policy_fixtures import (
    _FOLLOW,
    _HEADER,
    BFS_WITHOUT_COSTS,
    DEQUE_BFS,
    NESTED_RENAMED,
    PLAN_ONCE,
    REACTIVE,
    TIEBREAK_ASTAR,
)
from robocode.planning_explanations.search.policy_report import (
    _finding,
    save_policy_report,
)
from robocode.planning_explanations.search.policy_runner import (
    PolicyProgram,
    RunCapture,
    RunRecord,
    SearchEdit,
    Session,
    apply_edit,
    fresh_names,
    iteration_local,
    observe_episode,
    run_episode,
)
from robocode.planning_explanations.search.policy_worklist import (
    containers,
    worklist_signature,
)


def analyze(source, limits=PolicyLimits(), development=((0, 0),), validation=((0, 1),)):
    """Bank, program row and loop statuses (in source order) of one policy."""
    program, domain = PolicyProgram(source), DiscWorld()
    bank = discover_policy(program, domain, development, validation, limits)
    episode = validate_policy(program, domain, bank)["episodes"][0]
    order = sorted(bank.parts, key=lambda s: program.sites[s]["line"])
    return bank, episode, [bank.parts[s]["status"] for s in order]


def run_of(*frames) -> RunRecord:
    """A run whose loop checkpoints hold the given locals."""
    run = RunRecord("site", 0, ("reset",), RunCapture())
    for values in frames:
        run.capture.record("visit", values)
    return run


def nested_planner() -> str:
    """The same A*, with all search storage beyond the recorder's depth limit."""
    initialization = 'box = {"q": [(0.0, a)], "g": {a: 0.0}, "p": {a: None}}'
    source = NESTED_RENAMED.replace(
        initialization,
        "box = " + '{"layer": ' * 5 + initialization[6:] + "}" * 5,
    )
    for key in ("q", "g", "p"):
        source = source.replace(f'box["{key}"]', "box" + '["layer"]' * 5 + f'["{key}"]')
    return source


FOLLOW = "    def get_action(self, state):\n        return self._follow(state)"
NUMPY = "import math\nimport numpy as np\n"

# A harmless numpy tuple in the waypoint-following helper.
NUMPY_TUPLE = PLAN_ONCE.replace("import math\n", NUMPY).replace(
    '        x, y = state["robot"]\n        while self.path',
    '        x, y = state["robot"]\n'
    "        here = (np.float64(x), np.float64(y))\n"
    "        while self.path",
)

# The same A*, with its costs held as numpy numbers.
NUMPY_COSTS = PLAN_ONCE.replace("import math\n", NUMPY).replace(
    "nc = cost[cur] + math.hypot(di, dj)",
    "nc = np.float64(cost[cur] + math.hypot(di, dj))",
)

# plan_once plus a helper worklist loop the grammar cannot read (a countdown).
UNKNOWN_HELPER = PLAN_ONCE.replace(
    FOLLOW,
    "    def get_action(self, state):\n"
    "        work = [3]\n"
    "        while work:\n"
    "            n = work.pop()\n"
    "            if n > 0:\n"
    "                work.append(n - 1)\n"
    "        return self._follow(state)",
)

# An unreadable search that replaces the stored path in every action call.
UNKNOWN_REPLANNER = BFS_WITHOUT_COSTS.replace(
    FOLLOW,
    "    def get_action(self, state):\n"
    '        self.path = self._plan(state["robot"], self.goal, state["obstacles"])[1:]\n'
    "        return self._follow(state)",
)

APPEND_ONLY_BFS = (
    BFS_WITHOUT_COSTS.replace(
        "frontier = deque([root])", "frontier = [root]\n        cursor = 0"
    )
    .replace("while frontier:", "while cursor < len(frontier):")
    .replace(
        "cur = frontier.popleft()", "cur = frontier[cursor]\n            cursor += 1"
    )
)


# plan_once whose path rebuilding puts each point in a fresh list first.
POINT_LIST = PLAN_ONCE.replace(
    "            path.append(point(node))",
    "            pt = list(point(node))\n            path.append(pt)",
)

# plan_once with a 100 x 100 grid of integers next to its path rebuilding.
INTEGER_GRID = PLAN_ONCE.replace(
    "        path = []\n",
    "        grid = [[0] * 100 for _ in range(100)]\n        path = []\n",
)

# Breadth-first search one layer at a time: every round builds a new frontier.
_START = BFS_WITHOUT_COSTS.index("        frontier = deque([root])")
_END = BFS_WITHOUT_COSTS.index("        path = []")
LEVEL_BFS = BFS_WITHOUT_COSTS[:_START] + """        frontier = [root]
        parent = {root: None}
        while frontier and target not in parent:
            layer = []
            for cur in frontier:
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt = (cur[0] + di, cur[1] + dj)
                    if not free(nxt) or nxt in parent:
                        continue
                    parent[nxt] = cur
                    layer.append(nxt)
            frontier = layer
""" + BFS_WITHOUT_COSTS[_END:]


# plan_once with its A* written as a counted for loop that breaks when empty.
FOR_ASTAR = PLAN_ONCE.replace(
    "        while frontier:\n",
    "        for _ in range(100000):\n"
    "            if not frontier:\n"
    "                break\n",
)

# One A* per stop on the way to the goal, inside an outer for loop. Every stop
# rebuilds the search's containers and the path, so the outer loop carries none.
_STOPS = """
    def _plan(self, start, goal, obstacles):
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

        full = []
        for f in (0.3, 0.6, 1.0):
            stop = (start[0] + (goal[0] - start[0]) * f, start[1] + (goal[1] - start[1]) * f)
            root, target = cell(start), cell(stop)
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
                    nc = cost[cur] + 1.0
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
            full.extend(path)
            start = stop
        return full + [(goal[0], goal[1])]
"""
_PLAN_ON_RESET = """
    def reset(self, state, info):
        self.goal = tuple(state["goal"])
        self.path = self._plan(state["robot"], self.goal, state["obstacles"])

    def get_action(self, state):
        return self._follow(state)
"""
MULTI_STOP = _HEADER + _STOPS + _FOLLOW + _PLAN_ON_RESET

# A reactive policy that also fills a list from a counted for loop: the loop
# variable advances behind the growing list, but the iterator drives it.
LOOP_COUNTER = REACTIVE.replace(
    "        dx, dy = self.goal[0] - x, self.goal[1] - y\n",
    "        dx, dy = self.goal[0] - x, self.goal[1] - y\n"
    "        samples = []\n"
    "        for i in range(10):\n"
    "            samples.append(i * 0.1)\n",
)

# Dijkstra from scipy on a 4-connected grid graph; only the path walk is ours.
_SCIPY = """
    def _plan(self, start, goal, obstacles):
        step, n = 0.1, 25

        def free(i, j):
            x, y = i * step, j * step
            if not (0.05 <= x <= 2.45 and 0.05 <= y <= 2.45):
                return False
            return all(math.hypot(x - ox, y - oy) > r + 0.1 for ox, oy, r in obstacles)

        rows, cols = [], []
        for i in range(n):
            for j in range(n):
                for di, dj in ((1, 0), (0, 1)):
                    if i + di < n and j + dj < n and free(i, j) and free(i + di, j + dj):
                        rows += [i * n + j, (i + di) * n + j + dj]
                        cols += [(i + di) * n + j + dj, i * n + j]
        graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n * n, n * n))
        src = int(round(start[0] / step)) * n + int(round(start[1] / step))
        dst = int(round(goal[0] / step)) * n + int(round(goal[1] / step))
        _, pred = csgraph.dijkstra(graph, indices=src, return_predecessors=True)
        path, k = [], dst
        while k >= 0 and k != src:
            path.append(((k // n) * step, (k % n) * step))
            k = int(pred[k])
        path.reverse()
        return path + [(goal[0], goal[1])]
"""
SCIPY_DIJKSTRA = (
    _HEADER.replace(
        "import math\n",
        "import math\n\nimport numpy as np\n"
        "from scipy.sparse import csr_matrix, csgraph\n",
    )
    + _SCIPY
    + _FOLLOW
    + _PLAN_ON_RESET
)

# Breadth-first distances by growing a boolean reached grid one ring per round.
_WAVE = """
    def _plan(self, start, goal, obstacles):
        step, n = 0.1, 25
        axis = np.arange(n) * step
        cx, cy = np.meshgrid(axis, axis, indexing="ij")
        free = (cx >= 0.05) & (cx <= 2.45) & (cy >= 0.05) & (cy <= 2.45)
        for ox, oy, r in obstacles:
            free &= np.hypot(cx - ox, cy - oy) > r + 0.1
        dst = (int(round(goal[0] / step)), int(round(goal[1] / step)))
        reached = np.zeros((n, n), dtype=bool)
        reached[dst] = True
        self.dist = np.full((n, n), -1)
        self.dist[dst] = 0
        for k in range(1, 4 * n):
            grown = reached.copy()
            grown[1:, :] |= reached[:-1, :]
            grown[:-1, :] |= reached[1:, :]
            grown[:, 1:] |= reached[:, :-1]
            grown[:, :-1] |= reached[:, 1:]
            grown &= free
            if (grown == reached).all():
                break
            self.dist[grown & ~reached] = k
            reached = grown

    def reset(self, state, info):
        self.goal = tuple(state["goal"])
        self._plan(state["robot"], self.goal, state["obstacles"])

    def get_action(self, state):
        x, y = state["robot"]
        i, j = int(round(x / 0.1)), int(round(y / 0.1))
        best, here = self.goal, self.dist[i, j]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < 25 and 0 <= b < 25 and 0 <= self.dist[a, b] < here:
                best, here = (a * 0.1, b * 0.1), self.dist[a, b]
        return self._steer(x, y, best)
"""
WAVEFRONT = (
    _HEADER.replace("import math\n", "import math\n\nimport numpy as np\n")
    + _FOLLOW
    + _WAVE
)


# Backtracking depth-first search by recursion, sharing one path list. Its
# neighbour order runs into dead ends, so the path loses cells it had gained.
RECURSIVE_DFS = _HEADER + _FOLLOW + """
    def _dfs(self, cell, target, free, path, seen):
        path.append(cell)
        seen.add(cell)
        if cell == target:
            return True
        for di, dj in ((-1, 0), (0, -1), (1, 0), (0, 1)):
            nxt = (cell[0] + di, cell[1] + dj)
            if nxt not in seen and free(nxt) and self._dfs(nxt, target, free, path, seen):
                return True
        path.pop()
        return False

    def reset(self, state, info):
        self.goal = tuple(state["goal"])
        obstacles = state["obstacles"]

        def free(c):
            x, y = 0.1 * c[0], 0.1 * c[1]
            if not (0.05 <= x <= 2.45 and 0.05 <= y <= 2.45):
                return False
            return all(math.hypot(x - ox, y - oy) > r + 0.1 for ox, oy, r in obstacles)

        start = tuple(int(round(v / 0.1)) for v in state["robot"])
        target = tuple(int(round(v / 0.1)) for v in self.goal)
        path = []
        self._dfs(start, target, free, path, set())
        self.path = [(0.1 * i, 0.1 * j) for i, j in path] + [self.goal]

    def get_action(self, state):
        return self._follow(state)
"""

# A phase machine that re-dispatches by calling itself, and a linear recursion
# over a shrinking list of obstacles: both reactive.
REDISPATCH = REACTIVE.replace(
    "    def get_action(self, state):\n",
    "    def get_action(self, state):\n"
    "        if getattr(self, 'phase', 0) == 0:\n"
    "            self.phase = 1\n"
    "            return self.get_action(state)\n",
)
LINEAR_RECURSION = REACTIVE.replace(
    "    def get_action(self, state):\n",
    "    def _total(self, items):\n"
    "        if not items:\n"
    "            return 0.0\n"
    "        return items[0][2] + self._total(items[1:])\n\n"
    "    def get_action(self, state):\n"
    "        self.radius = self._total([list(o) for o in state['obstacles']])\n",
)


# Breadth-first search written as a for loop over the list it appends to.
FOR_QUEUE_BFS = _HEADER + """
    def _plan(self, start, goal, obstacles):
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
        queue = [root]
        parent = {root: None}
        for cur in queue:
            if cur == target:
                break
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cur[0] + di, cur[1] + dj)
                if nxt in parent or not free(nxt):
                    continue
                parent[nxt] = cur
                queue.append(nxt)
        path = []
        node = target if target in parent else None
        while node is not None:
            path.append(point(node))
            node = parent[node]
        path.reverse()
        return path + [(goal[0], goal[1])]
""" + _FOLLOW + _PLAN_ON_RESET

# A for loop that fills another list while iterating its own: not a queue.
OTHER_LIST = REACTIVE.replace(
    "        dx, dy = self.goal[0] - x, self.goal[1] - y\n",
    "        dx, dy = self.goal[0] - x, self.goal[1] - y\n"
    "        near = []\n"
    "        for o in state['obstacles']:\n"
    "            near.append(o)\n"
    "            near.append(o)\n",
)


# Plans "for 50 milliseconds": the number of rounds depends on the clock.
DEADLINE = REACTIVE.replace(
    "import math\n",
    "import math\nimport time\nfrom time import monotonic\n",
).replace(
    "        dx, dy = self.goal[0] - x, self.goal[1] - y\n",
    "        dx, dy = self.goal[0] - x, self.goal[1] - y\n"
    "        end, rounds = time.time() + 0.05, 0\n"
    "        while time.time() < end and monotonic() < end + 1:\n"
    "            rounds += 1\n"
    "        time.sleep(30)\n"
    "        self.rounds = rounds\n",
)


# An RRT whose tree lives in preallocated numpy arrays (invisible to snapshots),
# called from an outer retry loop, next to a loop that tries random goal points.
ARRAY_RRT = (
    _HEADER.replace("import math\n", "import math\n\nimport numpy as np\n") + """
    def _free(self, p, obstacles):
        if not (0.05 <= p[0] <= 2.45 and 0.05 <= p[1] <= 2.45):
            return False
        return all(math.hypot(p[0] - ox, p[1] - oy) > r + 0.1 for ox, oy, r in obstacles)

    def _rrt(self, start, goal, obstacles):
        nodes = np.zeros((600, 2))
        parent = np.full(600, -1)
        nodes[0], n = start, 1
        for _ in range(500):
            if self.rng.random() < 0.2:
                q = np.asarray(goal)
            else:
                q = self.rng.uniform(0.1, 2.4, size=2)
            i = int(np.argmin(np.hypot(*(nodes[:n] - q).T)))
            step = q - nodes[i]
            new = nodes[i] + step * min(1.0, 0.1 / max(math.hypot(*step), 1e-9))
            if not self._free(new, obstacles):
                continue
            nodes[n], parent[n] = new, i
            n += 1
            if math.hypot(*(new - goal)) < 0.1:
                break
        path, k = [], n - 1
        while k >= 0:
            path.append((float(nodes[k][0]), float(nodes[k][1])))
            k = parent[k]
        return path[::-1]

    def reset(self, state, info):
        self.rng = np.random.default_rng(0)
        self.goal = tuple(state["goal"])
        tries = []
        for _ in range(10):
            p = (float(self.rng.uniform(0.1, 2.4)), float(self.rng.uniform(0.1, 2.4)))
            if self._free(p, state["obstacles"]):
                tries.append(p)
        self.path = []
        for _ in range(2):
            path = self._rrt(np.asarray(state["robot"]), self.goal, state["obstacles"])
            if path:
                self.path = path + [self.goal]
                break

    def get_action(self, state):
        return self._follow(state)
""" + _FOLLOW
)


# Holds a generator it does not draw from in its object loop, and draws random
# numbers in another loop that only bumps a counter on itself.
RNG_BYSTANDER = (
    REACTIVE.replace("import math\n", "import math\n\nimport numpy as np\n")
    .replace(
        '        self.goal = tuple(state["goal"])\n',
        '        self.goal = tuple(state["goal"])\n'
        "        self.rng = np.random.default_rng(0)\n"
        "        self.seen, self.tries = [], 0\n",
    )
    .replace(
        "        dx, dy = self.goal[0] - x, self.goal[1] - y\n",
        "        dx, dy = self.goal[0] - x, self.goal[1] - y\n"
        "        for o in state['obstacles']:\n"
        "            self.seen.append(o)\n"
        "        for _ in range(3):\n"
        "            jitter = self.rng.normal()\n"
        "            self.tries += 1\n",
    )
)


class UnreadableLoopTest(unittest.TestCase):
    """A loop counts as zero only with evidence that it is not a search."""

    def assert_withheld(self, episode):
        """Unknown search effort is withheld and possible plan use stays open."""
        row = episode["accounting"]["program"]
        self.assertFalse(row["complete"])
        self.assertIsNone(row["expansion_events"])
        self.assertIsNone(row["planning_events"])
        # The followed plan may come from the unread loop: unknown, not zero.
        self.assertEqual(row["followed_plan_share"], 0.0)
        self.assertEqual(row["followed_plan_share_max"], 1.0)

    def test_a_cut_variable_list_is_unknown_not_zero(self):
        """Exhausting the selector budget cannot establish absence of search."""
        bank, episode, statuses = analyze(PLAN_ONCE, PolicyLimits(selectors=1))
        self.assertEqual(statuses[0], UNREADABLE)
        self.assertTrue(
            any(a.get("inventory_cut_runs") for a in bank.audit["parts"].values())
        )
        self.assert_withheld(episode)

    def test_a_too_shallow_inspection_is_unknown_not_zero(self):
        """A binding beyond the mapping depth still leaves worklist evidence."""
        _, episode, statuses = analyze(NESTED_RENAMED, PolicyLimits(mapping_depth=0))
        self.assertEqual(statuses[0], UNREADABLE)
        self.assert_withheld(episode)

    def test_storage_omitted_by_the_recorder_is_unknown(self):
        """No snapshot of the deep search container is not evidence of no search."""
        source = nested_planner()
        self.assertNotEqual(source, NESTED_RENAMED)
        bank, episode, statuses = analyze(source)
        self.assertEqual(statuses[0], UNREADABLE)
        self.assert_withheld(episode)
        evidence = next(iter(bank.audit["parts"].values()))
        self.assertEqual(evidence["unreadable"], "snapshot_omission")
        self.assertIn("box", evidence["worklist"]["omitted"])
        self.assertTrue(evidence["worklist"]["cut"])
        original = observe_episode(PolicyProgram(NESTED_RENAMED), DiscWorld(), (0, 1))
        nested = observe_episode(PolicyProgram(source), DiscWorld(), (0, 1))
        self.assertEqual(
            [c.action for c in original.calls], [c.action for c in nested.calls]
        )

    def test_append_only_queue_with_a_cursor_is_unknown(self):
        """BFS need not remove processed entries from its queue."""
        bank, episode, statuses = analyze(APPEND_ONLY_BFS)
        self.assertEqual(statuses[0], UNREADABLE)
        self.assert_withheld(episode)
        evidence = next(iter(bank.audit["parts"].values()))
        self.assertEqual(evidence["unreadable"], "growing_sequence_with_cursor")
        self.assertEqual(evidence["worklist"]["cursor"], "cursor")
        original = observe_episode(
            PolicyProgram(BFS_WITHOUT_COSTS), DiscWorld(), (0, 1)
        )
        indexed = observe_episode(PolicyProgram(APPEND_ONLY_BFS), DiscWorld(), (0, 1))
        self.assertEqual(
            [c.action for c in original.calls], [c.action for c in indexed.calls]
        )

    def test_a_search_outside_the_grammar_is_unknown_not_zero(self):
        """Breadth-first search without a cost map: no binding, but a worklist."""
        bank, episode, statuses = analyze(BFS_WITHOUT_COSTS)
        self.assertEqual(statuses[0], UNREADABLE)
        reasons = {a.get("unreadable") for a in bank.audit["parts"].values()}
        self.assertIn("worklist_without_binding", reasons)
        self.assert_withheld(episode)

    def test_a_level_by_level_search_stays_unknown(self):
        """A frontier replaced by a new list each round is still a worklist."""
        bank, episode, statuses = analyze(LEVEL_BFS)
        self.assertEqual(statuses[0], UNREADABLE)
        search = min(bank.parts, key=lambda site: int(site.split(":")[0]))
        self.assertEqual(
            bank.audit["parts"][search]["worklist"]["worklist"], "frontier"
        )
        self.assert_withheld(episode)

    def test_ordinary_loops_and_reactive_policies_still_count_as_zero(self):
        """The existing non-search controls keep their classifications."""
        _, episode, statuses = analyze(PLAN_ONCE)
        # Search, path reconstruction (only appends), waypoint popping (only pops).
        self.assertEqual(statuses, ["search", "not_search_shaped", "not_search_shaped"])
        self.assertTrue(episode["accounting"]["program"]["complete"])
        _, episode, statuses = analyze(REACTIVE)
        row = episode["accounting"]["program"]
        # Its obstacle loop is a watched for loop with nothing to carry.
        self.assertEqual(
            (statuses, row["complete"], row["planning_events"]),
            (["not_search_shaped"], True, 0),
        )

    def test_validation_rechecks_a_frozen_not_a_search_verdict(self):
        """A loop that only pops in development but keeps a worklist later."""
        source = REACTIVE.replace(
            '        self.goal = tuple(state["goal"])',
            '        self.goal = tuple(state["goal"])\n'
            '        work, rounds = list(state["obstacles"]), 0\n'
            "        while work:\n"
            "            item = work.pop()\n"
            "            rounds += 1\n"
            "            if rounds == 1 and work:\n"
            # A new entry: re-adding the popped one would leave no visible change.
            "                work.append(item[:2])",
        )
        _, episode, statuses = analyze(
            source, development=((1, 0),), validation=((3, 1),)
        )
        # The reset loop, then the reactive obstacle loop.
        self.assertEqual(statuses, ["not_search_shaped", "not_search_shaped"])
        (run,) = [r for r in episode["runs"] if r["part_status"] == UNREADABLE]
        self.assertEqual(run["worklist"]["worklist"], "work")
        self.assertFalse(episode["accounting"]["program"]["complete"])

    def test_validation_rechecks_new_snapshot_omissions(self):
        """A previously readable, shrinking list can become too deep to record."""
        source = REACTIVE.replace(
            '        self.goal = tuple(state["goal"])',
            '        self.goal = tuple(state["goal"])\n'
            '        work = ([1, 2] if len(state["obstacles"]) < 3 '
            "else [[[[[[1]]]]], 2])\n"
            "        while work:\n"
            "            work.pop()",
        )
        _, episode, statuses = analyze(
            source, development=((1, 0),), validation=((3, 1),)
        )
        self.assertEqual(statuses, ["not_search_shaped", "not_search_shaped"])
        self.assertFalse(episode["accounting"]["program"]["complete"])
        self.assertEqual(episode["runs"][0]["unreadable"], "snapshot_omission")

    def test_validation_only_search_has_an_unresolved_report(self):
        """First execution during validation must yield an unresolved headline."""
        store = (
            '        self.path = self._plan(state["robot"], self.goal, '
            'state["obstacles"])'
        )
        source = PLAN_ONCE.replace(
            store, '        if len(state["obstacles"]) >= 3:\n    ' + store
        )
        program, domain = PolicyProgram(source), DiscWorld()
        bank = discover_policy(program, domain, ((1, 0),), ((3, 100),))
        result = validate_policy(program, domain, bank)
        self.assertFalse(result["summary"]["complete"])
        self.assertIsNone(result["summary"]["mean"]["expansion_events"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = save_policy_report(
                root, "validation_only", program, bank, result, None
            )
            self.assertEqual(summary["finding"]["code"], "partially_explained")
            payload = json.loads((root / "report.json").read_text())
            label = payload["finding"]["label"]
            self.assertIn("unresolved", label)
            self.assertIn(label, (root / "report.md").read_text())
            self.assertIn(label, (root / "report.html").read_text())


class WiderFrontierTest(unittest.TestCase):
    """Common ways of writing the same searches are read, not left unknown."""

    def test_tie_break_numbers_in_heap_entries_are_read(self):
        """Tie-break bookkeeping preserves search support and plan attribution."""
        plain = analyze(PLAN_ONCE)[1]["accounting"]["program"]
        bank, episode, statuses = analyze(TIEBREAK_ASTAR)
        self.assertEqual(statuses[0], "search")
        self.assertEqual(
            {e.spec.layout for p in bank.parts.values() for e in p["explanations"]},
            {"tagged"},
        )
        row = episode["accounting"]["program"]
        self.assertTrue(row["complete"])
        self.assertEqual(row["expansion_events"], plain["expansion_events"])
        self.assertEqual(row["followed_plan_share"], 1.0)

    def test_a_deque_frontier_is_read(self):
        """Deque storage has a supported FIFO explanation."""
        bank, episode, statuses = analyze(DEQUE_BFS)
        self.assertEqual(statuses[0], "search")
        specs = {e.spec for p in bank.parts.values() for e in p["explanations"]}
        self.assertTrue(specs)
        self.assertTrue(
            all((s.layout, s.selection) == ("states", "fifo") for s in specs)
        )
        row = episode["accounting"]["program"]
        self.assertTrue(row["complete"])
        self.assertEqual(row["followed_plan_share"], 1.0)


class WorklistSignatureTest(unittest.TestCase):
    """The shape-free test behind "not a search"."""

    def test_a_container_that_gains_and_loses_is_a_worklist(self):
        """Queue churn is sufficient to withhold a no-search classification."""
        run = run_of({"q": [1, 2]}, {"q": [2, 3, 4]})
        self.assertEqual(worklist_signature(run)["worklist"], "q")

    def test_only_popping_or_only_appending_is_not_a_worklist(self):
        """One-direction mutations alone do not trigger worklist warnings."""
        for frames in (
            ({"q": [1, 2, 3]}, {"q": [2, 3]}, {"q": [3]}),
            ({"p": [1]}, {"p": [1, 2]}),
        ):
            self.assertIsNone(worklist_signature(run_of(*frames))["worklist"])

    def test_deques_sets_and_nested_containers_are_seen(self):
        """Inspection includes nested containers and reports depth cuts."""
        run = run_of({"box": {"q": deque([1])}}, {"box": {"q": deque([2])}})
        self.assertEqual(worklist_signature(run)["worklist"], 'box["q"]')
        found, cut = containers({"s": {1, 2}, "deep": [[[[[1]]]]]}, depth=2)
        self.assertIn("s", found)
        self.assertTrue(cut)

    def test_nested_cursor_is_not_name_dependent(self):
        """Nested sequence and integer paths are inventoried generically."""
        run = run_of(
            {"x": {"q": [10], "i": 0}},
            {"x": {"q": [10, 20], "i": 1}},
        )
        signature = worklist_signature(run)
        self.assertEqual(signature["worklist"], 'x["q"]')
        self.assertEqual(signature["cursor"], 'x["i"]')

    def test_a_counter_equal_to_the_length_is_not_a_cursor(self):
        """Smoothing appends one point per step; its index never trails the end."""
        run = run_of(
            {"result": [], "i": 0},
            {"result": [(0.1, 0.2)], "i": 1},
            {"result": [(0.1, 0.2), (0.3, 0.2)], "i": 2},
        )
        self.assertIsNone(worklist_signature(run)["worklist"])

    def test_a_frozen_or_out_of_range_integer_is_not_a_cursor(self):
        """Only an advancing integer inside the sequence can be a cursor."""
        for before, after in ((1, 1), (10, 11), (2, 1)):
            run = run_of({"q": [1], "i": before}, {"q": [1, 2], "i": after})
            self.assertIsNone(worklist_signature(run)["worklist"])

    def test_omitted_container_remains_visible_after_it_becomes_recordable(self):
        """A later readable value cannot erase an earlier observation gap."""
        run = run_of({"q": [[[[[[1]]]]]]}, {"q": [1]})
        signature = worklist_signature(run)
        self.assertTrue(signature["cut"])
        self.assertEqual(signature["omitted"], ["q"])

    def test_a_value_rebuilt_each_iteration_is_not_a_worklist(self):
        """A point replaced by a new list of the same size is a value."""
        run = run_of({"pt": [2, 3]}, {"pt": [2, 2]}, {"pt": [1, 2]})
        self.assertEqual(run.capture.replaced[1:], [{"pt"}, {"pt"}])
        self.assertIsNone(worklist_signature(run)["worklist"])

    def test_a_fixed_size_list_edited_in_place_can_be_a_worklist(self):
        """A preallocated queue keeps its size but is the same, edited list."""
        queue = [None, None]
        run = RunRecord("site", 0, ("reset",), RunCapture())
        run.capture.record("visit", {"q": queue})
        queue[0] = (1, 1)
        run.capture.record("visit", {"q": queue})
        queue[1] = (2, 2)
        queue[0] = None
        run.capture.record("visit", {"q": queue})
        self.assertEqual(run.capture.replaced[1:], [frozenset(), frozenset()])
        self.assertEqual(worklist_signature(run)["worklist"], "q")

    def test_a_replaced_list_that_changes_size_can_be_a_worklist(self):
        """A level-by-level frontier is replaced each round, with a new size."""
        run = run_of(
            {"frontier": [(0, 0)]},
            {"frontier": [(0, 1), (1, 0)]},
            {"frontier": [(0, 2), (1, 1), (2, 0)]},
        )
        self.assertEqual(worklist_signature(run)["worklist"], "frontier")

    def test_nested_containers_are_never_treated_as_values(self):
        """Replacement is only observed for variables; nested lists still count."""
        run = run_of({"box": {"pt": [2, 3]}}, {"box": {"pt": [2, 2]}})
        self.assertEqual(worklist_signature(run)["worklist"], 'box["pt"]')

    def test_integers_never_cut_inspection(self):
        """A grid of numbers is inspected fully, at any size or depth."""
        _, cut = containers({"grid": [[0] * 100 for _ in range(100)]})
        self.assertFalse(cut)
        _, cut = containers({"rows": [[1, 2]]}, depth=1)
        self.assertFalse(cut)
        _, cut = containers({"rows": [[[1, 2]]]}, depth=1)
        self.assertTrue(cut)

    def test_integers_inside_large_containers_are_still_cursors(self):
        """Lifting the integer limit keeps every integer a possible cursor."""
        run = run_of(
            {"q": [10], "marks": [0] * 5000},
            {"q": [10, 20], "marks": [1] + [0] * 4999},
        )
        signature = worklist_signature(run)
        self.assertEqual(signature["cursor"], "marks[0]")
        self.assertFalse(signature["cut"])

    def test_partly_opaque_containers_are_not_silently_absent(self):
        """Unsupported elements are followed by identity, or leave omission evidence."""
        run = run_of({"q": [object()]})
        self.assertEqual(run.capture.members[0], {"q": (1, True, False, True)})
        inner = [object()]
        run = run_of({"q": [inner]})
        inner.append(object())
        run.capture.record("visit", {"q": [inner]})
        signature = worklist_signature(run)
        self.assertTrue(signature["cut"])
        self.assertEqual(signature["omitted"], ["q"])


class DequeAndTaggedEditTest(unittest.TestCase):
    """Live edits on deques and on entries with tie-break numbers."""

    def test_edits_address_entries_by_priority_and_cell(self):
        """Live edits preserve the heap/deque representation contract."""
        heap = [(0.5, 3, (1, 2)), (0.7, 4, (3, 4)), (0.5, 9, (1, 2))]
        apply_edit({"f": heap}, SearchEdit("s", 0, 0, "drop", "f", entry=(0.5, (1, 2))))
        self.assertEqual(
            sorted((e[0], e[-1]) for e in heap), [(0.5, (1, 2)), (0.7, (3, 4))]
        )
        apply_edit({"f": heap}, SearchEdit("s", 0, 0, "drop_cell", "f", cell=(1, 2)))
        self.assertEqual(heap, [(0.7, 4, (3, 4))])
        queue = deque([(1, 1), (2, 2), (3, 3)])
        apply_edit(
            {"q": queue},
            SearchEdit("s", 0, 0, "drop_cell", "q", cell=(2, 2), heap=False),
        )
        self.assertEqual(queue, deque([(1, 1), (3, 3)]))
        apply_edit({"q": queue}, SearchEdit("s", 0, 0, "clear", "q", heap=False))
        self.assertEqual(len(queue), 0)

    def test_deque_and_set_locals_are_recorded(self):
        """Snapshots preserve queue order, length bounds and set contents."""
        run = run_of({"q": deque([(1, 2)], 5), "seen": {(1, 2)}})
        _, values = next(run.capture.frames())
        self.assertEqual(values["q"], deque([(1, 2)], 5))
        self.assertEqual(values["q"].maxlen, 5)
        self.assertEqual(values["seen"], {(1, 2)})


class NumpyAndUnknownLoopTest(unittest.TestCase):
    """numpy values are numbers; an unknown loop does not erase plan evidence."""

    def test_numpy_numbers_in_a_helper_do_not_make_loops_unknown(self):
        """A tuple of numpy numbers is recorded as numbers, not omitted state."""
        self.assertIn("np.float64(x)", NUMPY_TUPLE)
        _, episode, statuses = analyze(NUMPY_TUPLE)
        self.assertEqual(statuses, ["search", "not_search_shaped", "not_search_shaped"])
        row = episode["accounting"]["program"]
        self.assertTrue(row["complete"])
        self.assertEqual(row["followed_plan_share"], 1.0)

    def test_numpy_costs_are_read_as_numbers(self):
        """A search keeping numpy costs is explained like the plain one."""
        self.assertIn("np.float64(cost[cur]", NUMPY_COSTS)
        plain = analyze(PLAN_ONCE)[1]["accounting"]["program"]
        _, episode, statuses = analyze(NUMPY_COSTS)
        self.assertEqual(statuses[0], "search")
        row = episode["accounting"]["program"]
        self.assertEqual(row["expansion_events"], plain["expansion_events"])

    def test_an_unknown_helper_loop_keeps_plan_evidence(self):
        """Totals are withheld, but proven plan-following steps stay proven."""
        _, episode, statuses = analyze(UNKNOWN_HELPER)
        self.assertEqual(statuses[0], "search")
        self.assertEqual(statuses[-1], UNREADABLE)
        row = episode["accounting"]["program"]
        self.assertFalse(row["complete"])
        self.assertIsNone(row["expansion_events"])
        self.assertEqual(row["followed_plan_share"], 1.0)
        self.assertEqual(row["unresolved_loop"], 0)

    def test_a_plan_stored_beside_an_unknown_loop_is_not_proven(self):
        """An unknown loop that may replace the plan in the same call gets no credit."""
        _, episode, statuses = analyze(UNKNOWN_REPLANNER)
        self.assertEqual(statuses[0], UNREADABLE)
        row = episode["accounting"]["program"]
        self.assertEqual(row["followed_plan_share"], 0.0)
        self.assertEqual(row["followed_plan_share_max"], 1.0)
        self.assertGreater(row["unresolved_loop"], 0)

    def test_small_numpy_arrays_and_tuples_of_lists_are_recorded(self):
        """Waypoint vectors and records of lists are values, not snapshot gaps."""
        capture = RunCapture()
        out = [np.array([0.5, 1.0]), np.array([0.6, 1.0])]
        check = ([(np.array([1.0, 2.0]), 0.5)], None, 0.1)
        capture.record(
            "visit", {"out": out, "check": check, "big": np.zeros(500), "p": out[0]}
        )
        _, values = next(capture.frames())
        self.assertEqual(values["out"], [(0.5, 1.0), (0.6, 1.0)])
        self.assertEqual(values["check"], ([((1.0, 2.0), 0.5)], None, 0.1))
        self.assertEqual(values["p"], (0.5, 1.0))
        self.assertNotIn("big", values)
        self.assertEqual(capture.omitted_containers, set())

    def test_a_worklist_inside_a_tuple_is_still_seen(self):
        """Recording a tuple of lists must not hide the queue it carries."""
        run = run_of({"state": ([1, 2], "x")}, {"state": ([2, 3, 4], "x")})
        self.assertEqual(worklist_signature(run)["worklist"], "state[0]")

    def test_only_mutable_unrecorded_state_counts_as_omitted(self):
        """Tuples of numbers or objects cannot hide a worklist; lists can."""
        capture = RunCapture()
        capture.record(
            "visit",
            {
                "here": (np.float64(1.5), np.int64(2)),
                "node": (1.0, object()),
                "heap": [(0.5, object())],
                "box": ([object()], 2),
            },
        )
        _, values = next(capture.frames())
        self.assertEqual(values["here"], (1.5, 2))
        self.assertIs(type(values["here"][0]), float)
        # A list of object-holding tuples is followed by item identity instead,
        # and a tuple holding a list counts as omitted once that list changes.
        self.assertEqual(capture.omitted_containers, set())
        self.assertEqual(capture.members[0]["heap"], (1, True, False, True))
        capture.record("visit", {"box": ([object()], 2)})
        self.assertEqual(capture.omitted_containers, {"box"})


class OrdinaryCodeKeepsTotalsTest(unittest.TestCase):
    """Common non-search code next to a search does not hide its totals."""

    def test_a_point_rebuilt_as_a_list_keeps_totals(self):
        """A fresh [x, y] list per path point is not a to-do list."""
        plain = analyze(PLAN_ONCE)[1]["accounting"]["program"]
        _, episode, statuses = analyze(POINT_LIST)
        self.assertEqual(statuses, ["search", "not_search_shaped", "not_search_shaped"])
        row = episode["accounting"]["program"]
        self.assertTrue(row["complete"])
        self.assertEqual(row["expansion_events"], plain["expansion_events"])

    def test_a_large_integer_grid_keeps_totals(self):
        """Ten thousand integers are not a reason for "could not inspect"."""
        plain = analyze(PLAN_ONCE)[1]["accounting"]["program"]
        _, episode, statuses = analyze(INTEGER_GRID)
        self.assertEqual(statuses, ["search", "not_search_shaped", "not_search_shaped"])
        row = episode["accounting"]["program"]
        self.assertTrue(row["complete"])
        self.assertEqual(row["expansion_events"], plain["expansion_events"])


class ReportBoundsTest(unittest.TestCase):
    """Headlines name only what ran; withheld totals show their known part."""

    def test_an_unreadable_loop_that_never_ran_does_not_withhold(self):
        """The headline follows validation, with a note about the idle loop."""
        bank = SimpleNamespace(
            parts={"3:4": {"status": UNREADABLE}, "9:8": {"status": "search"}}
        )
        runs = [{"site": "9:8", "part_status": "search"}]
        validation = {"episodes": [{"runs": runs}], "summary": {"complete": True}}
        finding = _finding(bank, validation)
        self.assertEqual(finding["code"], "search_supported")
        self.assertIn("did not run in validation", finding["note"])
        runs.append({"site": "3:4", "part_status": UNREADABLE})
        finding = _finding(bank, {**validation, "summary": {"complete": False}})
        self.assertEqual(finding["code"], "unreadable_loops")
        self.assertNotIn("note", finding)

    def test_withheld_totals_are_shown_as_lower_bounds(self):
        """An unknown loop can only add effort, so the known part is "≥"."""
        program, domain = PolicyProgram(UNKNOWN_HELPER), DiscWorld()
        bank = discover_policy(program, domain, ((0, 0),), ((0, 1),))
        result = validate_policy(program, domain, bank)
        row = result["episodes"][0]["accounting"]["program"]
        known = row["explained_only_expansion_events"]
        self.assertGreater(known, 0)
        self.assertIsNone(row["initialization_expansion_events"])
        self.assertEqual(row["explained_only_initialization_expansion_events"], known)
        self.assertEqual(result["summary"]["lower_bound"]["expansion_events"], known)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = save_policy_report(root, "helper", program, bank, result, None)
            self.assertIsNone(summary["mean_expansion_events"])
            self.assertEqual(summary["lower_bound_expansion_events"], known)
            # The search score is withheld too, with its known part as a bound.
            self.assertIsNone(summary["mean_search_score"])
            self.assertGreater(summary["lower_bound_search_score"], 0)
            self.assertIn("Search score (per call)", (root / "report.md").read_text())
            self.assertIn(f"| ≥ {known} |", (root / "report.md").read_text())


def loop_of(source: str):
    """The first loop statement of a code fragment."""
    return next(
        n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.For, ast.While))
    )


def grids(*rows_of_cells, size=7):
    """Boolean grids with the given True cells."""
    result = []
    for cells in rows_of_cells:
        grid = np.zeros((size, size), dtype=bool)
        for cell in cells:
            grid[cell] = True
        result.append({"reached": grid})
    return result


class WatchedForLoopTest(unittest.TestCase):
    """For loops are parts; only what one iteration hands the next is a worklist."""

    def test_a_for_loop_search_is_read_like_a_while_loop(self):
        """The same A* written as a counted for loop gets the same numbers."""
        _, episode, statuses = analyze(
            FOR_ASTAR, development=((1, 0), (3, 1)), validation=((3, 100),)
        )
        _, same, _ = analyze(
            PLAN_ONCE, development=((1, 0), (3, 1)), validation=((3, 100),)
        )
        self.assertEqual(statuses[0], "search")
        row, expected = episode["accounting"]["program"], same["accounting"]["program"]
        self.assertTrue(row["complete"])
        for key in ("search_score", "expansion_events", "planning_events"):
            self.assertEqual(row[key], expected[key])

    def test_an_outer_loop_around_a_search_is_not_a_worklist(self):
        """Per-stop containers are rebuilt each iteration: nothing is carried."""
        program = PolicyProgram(MULTI_STOP)
        outer = next(s for s, i in program.sites.items() if i.get("loop") == "for")
        self.assertTrue(
            {"frontier", "cost", "parent", "path"} <= set(program.sites[outer]["fresh"])
        )
        _, episode, statuses = analyze(
            MULTI_STOP, development=((1, 0), (3, 1)), validation=((3, 100),)
        )
        self.assertEqual(statuses[:2], ["not_search_shaped", "search"])
        self.assertTrue(episode["accounting"]["program"]["complete"])
        self.assertGreater(episode["accounting"]["program"]["search_score"], 0)

    def test_iteration_local_variables(self):
        """Rebound before every read in an iteration (even inside a branch)."""
        cases = {
            # Astra's candidate loop: a carry path rebuilt in a branch, then read.
            "for obj in ordered:\n"
            "    if obj:\n"
            "        delivery = route(obj)\n"
            "        if delivery is None:\n"
            "            continue\n"
            "        choices.append((len(delivery), obj))\n": {"delivery", "obj"},
            # Level-by-level BFS: the frontier is read before it is rebound.
            "while frontier:\n"
            "    layer = []\n"
            "    for cur in frontier:\n"
            "        layer.append(cur)\n"
            "    frontier = layer\n": {"cur", "layer"},
            # Rebound in one branch only, read after it.
            "for x in xs:\n"
            "    if x:\n"
            "        path = plan(x)\n"
            "    use(path)\n": {"x"},
            # An alias of outside state is still inspected.
            "while True:\n    q = self.open\n    node = pop(q)\n": {"node"},
        }
        for source, expected in cases.items():
            self.assertEqual(iteration_local(loop_of(source)), expected, source)

    def test_a_counted_loop_variable_is_not_a_read_cursor(self):
        """`for i in range(n): xs.append(...)` stays zero."""
        _, episode, statuses = analyze(LOOP_COUNTER)
        self.assertEqual(statuses, ["not_search_shaped", "not_search_shaped"])
        row = episode["accounting"]["program"]
        self.assertEqual((row["complete"], row["planning_events"]), (True, 0))

    def test_fresh_names(self):
        """For variables and values rebuilt before any use carry nothing."""
        source = (
            "for goal in goals:\n"
            "    path = plan(goal)\n"
            "    seen = set()\n"
            "    total = total + len(path)\n"
            "    queue = [goal] + queue\n"
            "    alias = work\n"
            "    items[0] = []\n"
            "    for a, (b, c) in pairs:\n"
            "        pass\n"
            "    if path:\n"
            "        late = []\n"
            "    used_first = len(late)\n"
        )
        fresh = fresh_names(loop_of(source))
        self.assertEqual(fresh, ["a", "b", "c", "goal", "path", "seen", "used_first"])
        # A while loop's test reads its variables before the body rebuilds them.
        fresh = fresh_names(
            loop_of("while frontier:\n    frontier = []\n    x = f()\n")
        )
        self.assertEqual(fresh, ["x"])


class IteratedQueueTest(unittest.TestCase):
    """A for loop over a list it appends to reads it with its own cursor."""

    def test_a_for_loop_queue_is_unknown_not_zero(self):
        """Breadth-first search as `for cur in queue: queue.append(...)`."""
        program = PolicyProgram(FOR_QUEUE_BFS)
        bank = discover_policy(program, DiscWorld(), ((1, 0),), ((3, 100),))
        result = validate_policy(program, DiscWorld(), bank)
        (site,) = [s for s, i in program.sites.items() if i.get("iterates") == "queue"]
        self.assertEqual(bank.parts[site]["status"], UNREADABLE)
        self.assertEqual(bank.audit["parts"][site]["unreadable"], "iterated_list_grows")
        self.assertEqual(_finding(bank, result)["code"], "unreadable_loops")

    def test_growing_another_list_stays_zero(self):
        """Only the list being iterated counts."""
        _, episode, statuses = analyze(OTHER_LIST)
        self.assertEqual(statuses, ["not_search_shaped", "not_search_shaped"])
        self.assertTrue(episode["accounting"]["program"]["complete"])

    def test_signature(self):
        """Growth of the iterated list counts; a rebound name does not."""
        queue = [1]
        run = run_of()
        for item in (2, 3):
            run.capture.record("visit", {"queue": queue})
            queue.append(item)
        run.capture.record("exit", {"queue": queue})
        self.assertIsNone(worklist_signature(run)["worklist"])
        signature = worklist_signature(run, iterated="queue")
        self.assertEqual(signature["kind"], "iterated_list_grows")
        rebound = run_of({"queue": [1]}, {"queue": [1, 2]})
        self.assertIsNone(worklist_signature(rebound, iterated="queue")["worklist"])

    def test_what_a_for_loop_iterates(self):
        """A plain variable, or enumerate of one; nothing else."""
        source = (
            "class GeneratedApproach:\n"
            "    def a(self, q):\n"
            "        for x in q:\n"
            "            pass\n"
            "    def b(self, q):\n"
            "        for i, x in enumerate(q):\n"
            "            pass\n"
            "    def c(self, q):\n"
            "        for x in list(q):\n"
            "            pass\n"
        )
        sites = PolicyProgram(source).sites.values()
        self.assertEqual(sorted(str(i["iterates"]) for i in sites), ["None", "q", "q"])


class FakeClockTest(unittest.TestCase):
    """A time budget is a fixed number of clock reads, recorded or not."""

    def test_a_deadline_loop_behaves_the_same_when_recorded(self):
        """Same actions with and without recording, without really sleeping."""
        program = PolicyProgram(DEADLINE)
        started = time.monotonic()
        trace = observe_episode(program, DiscWorld(), (0, 0))
        self.assertLess(time.monotonic() - started, 30)
        (site,) = [s for s, i in program.sites.items() if i.get("loop") == "while"]
        visits = {r.visits for c in trace.calls for r in c.runs if r.site == site}
        # 0.05 s at 1 ms per read, two reads per round: 25 rounds, every call.
        self.assertEqual(visits, {26})

    def test_a_reexecuted_call_starts_from_its_own_clock(self):
        """Probes replay a call at its original time and leave the episode's."""
        program = PolicyProgram(DEADLINE)
        trace = run_episode(program, DiscWorld(), (0, 0))
        call = trace.calls[3]
        policy = program.policy_class()(None, None, None)
        policy.goal = (1.0, 1.0)
        program.clock.now = 5.0
        session = Session(capture=False, clock=call.clock)
        program.call(
            policy, "get_action", ({"robot": (0.5, 0.5), "obstacles": []},), session
        )
        self.assertEqual(program.clock.now, 5.0)
        self.assertEqual(policy.rounds, 25)
        self.assertGreater(trace.calls[3].clock, trace.calls[2].clock)


class RandomHiddenStateTest(unittest.TestCase):
    """Random draws while state the snapshots do not hold changes: maybe a search."""

    @classmethod
    def setUpClass(cls):
        cls.program = PolicyProgram(ARRAY_RRT)
        cls.bank = discover_policy(cls.program, DiscWorld(), ((1, 0),), ((3, 100),))

    def part(self, text):
        """The part whose first line contains `text`."""
        (site,) = [s for s, i in self.program.sites.items() if text in i["source"]]
        return self.bank.parts[site]["status"], self.bank.audit["parts"][site]

    def test_an_rrt_in_numpy_arrays_is_unknown_not_zero(self):
        """Its nodes and parents are arrays the recorder does not copy."""
        status, audit = self.part("for _ in range(500)")
        self.assertEqual(status, UNREADABLE)
        self.assertEqual(audit["unreadable"], "random_with_hidden_state")
        self.assertIn(audit["worklist"]["worklist"], ("nodes", "parent"))

    def test_the_loop_that_calls_the_planner_is_not_flagged(self):
        """The planner's draws belong to the planner's own loop."""
        status, _ = self.part("for _ in range(2)")
        self.assertEqual(status, "not_search_shaped")

    def test_holding_a_generator_or_bumping_a_counter_is_not_flagged(self):
        """Finding `self.rng` is not a draw; a counter cannot hold a tree."""
        _, episode, statuses = analyze(RNG_BYSTANDER)
        self.assertNotIn(UNREADABLE, statuses)
        self.assertTrue(episode["accounting"]["program"]["complete"])

    def test_trying_random_points_with_visible_results_is_not_flagged(self):
        """Random goal points kept in a visible list: the usual tests decide."""
        status, _ = self.part("for _ in range(10)")
        self.assertEqual(status, "not_search_shaped")


class LibrarySearchTest(unittest.TestCase):
    """Library graph searches are parts, and always unknown."""

    def test_imports_resolve_to_library_searches(self):
        """Aliases and from-imports are followed; non-search helpers are not parts."""
        source = (
            "import networkx as nx\n"
            "import scipy.sparse.csgraph\n"
            "from scipy.sparse.csgraph import shortest_path as sp\n"
            "from scipy import ndimage\n"
            "class GeneratedApproach:\n"
            "    def plan(self, g, grid):\n"
            "        a = nx.astar_path(g, 0, 1)\n"
            "        b = scipy.sparse.csgraph.dijkstra(g)\n"
            "        c = sp(g)\n"
            "        d = ndimage.label(grid)\n"
            "        e = nx.Graph()\n"
            "        return a, b, c, d, e\n"
        )
        calls = sorted(
            info["call"]
            for info in PolicyProgram(source).sites.values()
            if info["kind"] == "library"
        )
        self.assertEqual(
            calls,
            [
                "networkx.astar_path",
                "scipy.sparse.csgraph.dijkstra",
                "scipy.sparse.csgraph.shortest_path",
            ],
        )

    def test_a_library_search_is_unknown_not_zero(self):
        """scipy's Dijkstra withholds totals; the policy's own loops stay zero."""
        program, domain = PolicyProgram(SCIPY_DIJKSTRA), DiscWorld()
        bank = discover_policy(program, domain, ((1, 0),), ((3, 100),))
        result = validate_policy(program, domain, bank)
        (site,) = [s for s, i in program.sites.items() if i["kind"] == "library"]
        self.assertEqual(bank.parts[site]["status"], UNREADABLE)
        self.assertEqual(bank.audit["parts"][site]["unreadable"], LIBRARY)
        others = {p["status"] for s, p in bank.parts.items() if s != site}
        self.assertLessEqual(
            others, {"not_search_shaped", "not_executed_in_development"}
        )
        self.assertEqual(_finding(bank, result)["code"], "unreadable_loops")
        self.assert_withheld(result["episodes"][0])

    assert_withheld = UnreadableLoopTest.assert_withheld


class WavefrontTest(unittest.TestCase):
    """A boolean grid growing by its own neighbours is a worklist in grid form."""

    def signature(self, *frames, ignore=frozenset()):
        """The worklist signature of one run with these frames."""
        return worklist_signature(run_of(*frames), ignore=ignore)

    def test_three_neighbour_growth_steps_are_a_wavefront(self):
        """Three rings grown from a seed, and nothing else changed."""
        frames = grids(
            [(3, 3)],
            [(3, 3), (3, 4)],
            [(3, 3), (3, 4), (4, 5)],
            [(3, 3), (3, 4), (4, 5), (5, 6)],
        )
        signature = self.signature(*frames)
        self.assertEqual(
            (signature["worklist"], signature["kind"]),
            ("reached", "growing_reached_grid"),
        )
        # A wavefront in a variable that cannot carry it is not this loop's.
        self.assertIsNone(
            self.signature(*frames, ignore=frozenset({"reached"}))["worklist"]
        )

    def test_short_growth_stamps_and_shrinking_are_not(self):
        """Too short, stamped shapes far apart, or cells removed."""
        twice = grids([(3, 3)], [(3, 3), (3, 4)], [(3, 3), (3, 4), (3, 5)])
        stamps = grids([(0, 0)], [(0, 0), (3, 3)], [(0, 0), (3, 3), (6, 6)])
        shrink = grids([(3, 3), (3, 4)], [(3, 3)], [(3, 3), (3, 4)], [(3, 3)])
        for frames in (twice, stamps, shrink):
            self.assertIsNone(self.signature(*frames)["worklist"])

    def test_bands_then_a_stamp_are_not_a_wavefront(self):
        """An obstacle map thickened by thin bands, then stamped (Astra's map)."""
        frames = grids(
            [(0, c) for c in range(7)],
            [(r, c) for r in range(2) for c in range(7)],
            [(r, c) for r in range(3) for c in range(7)],
            [(r, c) for r in range(4) for c in range(7)],
            [(r, c) for r in range(4) for c in range(7)] + [(6, 6)],
        )
        self.assertIsNone(self.signature(*frames)["worklist"])

    def test_a_wavefront_policy_is_unknown_not_zero(self):
        """A breadth-first distance field kept as a reached grid."""
        _, episode, statuses = analyze(
            WAVEFRONT, development=((1, 0),), validation=((3, 100),)
        )
        self.assertIn(UNREADABLE, statuses)
        program = PolicyProgram(WAVEFRONT)
        bank = discover_policy(program, DiscWorld(), ((1, 0),), ((3, 100),))
        reasons = {r.get("unreadable") for r in bank.audit["parts"].values()}
        self.assertIn("growing_reached_grid", reasons)
        self.assertFalse(episode["accounting"]["program"]["complete"])


class RecursionTest(unittest.TestCase):
    """A function that calls itself is a part; its calls are its iterations."""

    def statuses(self, source):
        """Bank, validation result and the recursion part of one policy."""
        program = PolicyProgram(source)
        bank = discover_policy(program, DiscWorld(), ((1, 0),), ((3, 100),))
        result = validate_policy(program, DiscWorld(), bank)
        (site,) = [s for s, i in program.sites.items() if i["kind"] == "recursion"]
        return bank, result, site

    def test_backtracking_recursion_is_unknown_not_zero(self):
        """The shared path gains and loses cells across calls: a worklist."""
        bank, result, site = self.statuses(RECURSIVE_DFS)
        self.assertEqual(bank.parts[site]["status"], UNREADABLE)
        self.assertEqual(bank.audit["parts"][site]["worklist"]["worklist"], "path")
        self.assertEqual(_finding(bank, result)["code"], "unreadable_loops")

    def test_reactive_recursion_stays_zero(self):
        """Re-dispatching, and recursing on a shrinking list, carry no worklist."""
        for source in (REDISPATCH, LINEAR_RECURSION):
            bank, result, site = self.statuses(source)
            self.assertEqual(bank.parts[site]["status"], "not_search_shaped")
            row = result["episodes"][0]["accounting"]["program"]
            self.assertEqual((row["complete"], row["planning_events"]), (True, 0))

    def test_one_run_per_outermost_call(self):
        """Nested calls are visits of the outermost call's run."""
        program = PolicyProgram(RECURSIVE_DFS)
        trace = observe_episode(program, DiscWorld(), (1, 0))
        runs = [r for c in trace.calls for r in c.runs if r.site.endswith("Recursion")]
        self.assertEqual(len(runs), 1)
        self.assertGreater(runs[0].visits, 10)
        self.assertTrue(runs[0].exited)
        self.assertEqual(runs[0].depth, 0)


class ShiftedPositionTest(unittest.TestCase):
    """A list's elements are compared by position only if its length is kept."""

    def test_a_shrinking_list_of_lists_is_not_churn(self):
        """After `items = items[1:]`, items[0] is the next element."""
        run = run_of(
            {"items": [[1, 2], [3, 4], [5, 6]]},
            {"items": [[3, 4], [5, 6]]},
            {"items": [[5, 6]]},
        )
        self.assertIsNone(worklist_signature(run)["worklist"])

    def test_a_bucket_queue_is_still_churn(self):
        """Buckets keep their positions; one gains and loses items in place."""
        run = run_of(
            {"buckets": [[1], [2]]},
            {"buckets": [[], [2, 5]]},
            {"buckets": [[7], [5]]},
        )
        self.assertEqual(worklist_signature(run)["worklist"], "buckets[0]")


class OpaqueItemTest(unittest.TestCase):
    """Containers of objects are followed by item identity, not omitted."""

    def test_an_integer_the_loop_never_reads_is_not_a_cursor(self):
        """Astra's `components, _ = label(...)`: `_` grows but indexes nothing."""
        objects = [object() for _ in range(4)]
        choices: list = []
        run = run_of()
        for count, obj in enumerate(objects):
            run.capture.record("visit", {"choices": choices, "_": count})
            choices.extend([(1.0, obj), (2.0, obj)])
        self.assertEqual(worklist_signature(run)["cursor"], "_")
        signature = worklist_signature(run, read=frozenset({"choices"}))
        self.assertIsNone(signature["worklist"])
        program = PolicyProgram(
            "class GeneratedApproach:\n"
            "    def plan(self, ordered):\n"
            "        choices = []\n"
            "        for obj in ordered:\n"
            "            if obj:\n"
            "                labels, _ = obj.label()\n"
            "            choices.append((labels, obj))\n"
            "        return choices\n"
        )
        sites = list(program.sites.values())
        self.assertEqual(len(sites), 1)
        site = sites[0]
        self.assertNotIn("_", site["read"])
        self.assertIn("labels", site["read"])

    def test_a_list_of_objects_only_iterated_or_appended_is_readable(self):
        """`for obj in objects: choices.append((score, obj))` is not a worklist."""
        objects = [object() for _ in range(3)]
        choices: list = []
        run = run_of()
        for obj in objects:
            run.capture.record(
                "visit", {"objects": objects, "choices": choices, "obj": obj}
            )
            choices.append((1.0, obj))
        run.capture.record("exit", {"objects": objects, "choices": choices})
        signature = worklist_signature(run)
        self.assertEqual((signature["worklist"], signature["omitted"]), (None, []))

    def test_a_queue_of_objects_still_churns(self):
        """Popping one node object and pushing another is a worklist."""
        nodes = [object() for _ in range(4)]
        queue = deque(nodes[:2])
        run = run_of()
        for node in nodes[2:]:
            run.capture.record("visit", {"queue": queue})
            queue.popleft()
            queue.append(node)
        run.capture.record("visit", {"queue": queue})
        self.assertEqual(worklist_signature(run)["worklist"], "queue")

    def test_lists_inside_are_omitted_once_they_change(self):
        """Lists of lists of objects could hide a worklist one level down."""
        buckets = [[object()], [object()]]
        run = run_of()
        run.capture.record("visit", {"buckets": buckets, "state": {"block": object()}})
        run.capture.record("visit", {"buckets": buckets, "state": {"block": object()}})
        # Read-only so far: nothing inside was gained or lost.
        self.assertEqual(worklist_signature(run)["omitted"], [])
        buckets[0].append(buckets[1].pop())
        run.capture.record("visit", {"buckets": buckets, "state": {"block": object()}})
        self.assertEqual(worklist_signature(run)["omitted"], ["buckets"])


if __name__ == "__main__":
    unittest.main()
