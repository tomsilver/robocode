"""Issue 2: a loop the analyzer cannot read is unknown, never zero planning."""

import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robocode.planning_explanations.search.policy_discovery import (
    UNREADABLE,
    PolicyLimits,
    discover_policy,
    validate_policy,
)
from robocode.planning_explanations.search.policy_domains import DiscWorld
from robocode.planning_explanations.search.policy_fixtures import (
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
    apply_edit,
    observe_episode,
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
        self.assertEqual(
            (statuses, row["complete"], row["planning_events"]), ([], True, 0)
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
        self.assertEqual(statuses, ["not_search_shaped"])
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
        self.assertEqual(statuses, ["not_search_shaped"])
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
        """Unsupported elements also leave omission evidence."""
        run = run_of({"q": [object()]})
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
        self.assertEqual(capture.omitted_containers, {"heap", "box"})


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
            self.assertIn(f"| ≥ {known} |", (root / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
