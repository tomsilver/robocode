"""Issue 2: a loop the analyzer cannot read is unknown, never zero planning."""

import json
import tempfile
import unittest
from collections import deque
from pathlib import Path

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
from robocode.planning_explanations.search.policy_report import save_policy_report
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
            "                work.append(item)",
        )
        _, episode, statuses = analyze(
            source, development=((1, 0),), validation=((3, 1),)
        )
        self.assertEqual(statuses, ["not_search_shaped"])
        self.assertTrue(any(r["part_status"] == UNREADABLE for r in episode["runs"]))
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


if __name__ == "__main__":
    unittest.main()
