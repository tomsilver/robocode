"""Issue 2: a loop the analyzer cannot read is unknown, never zero planning."""

import unittest
from collections import deque

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
from robocode.planning_explanations.search.policy_runner import (
    PolicyProgram,
    RunCapture,
    RunRecord,
    SearchEdit,
    apply_edit,
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


class UnreadableLoopTest(unittest.TestCase):
    """A loop counts as zero only with evidence that it is not a search."""

    def assert_withheld(self, episode):
        row = episode["accounting"]["program"]
        self.assertFalse(row["complete"])
        self.assertIsNone(row["expansion_events"])
        self.assertIsNone(row["planning_events"])
        # The followed plan may come from the unread loop: unknown, not zero.
        self.assertEqual(row["followed_plan_share"], 0.0)
        self.assertEqual(row["followed_plan_share_max"], 1.0)

    def test_a_cut_variable_list_is_unknown_not_zero(self):
        bank, episode, statuses = analyze(PLAN_ONCE, PolicyLimits(selectors=1))
        self.assertEqual(statuses[0], UNREADABLE)
        self.assertTrue(
            any(a.get("inventory_cut_runs") for a in bank.audit["parts"].values())
        )
        self.assert_withheld(episode)

    def test_a_too_shallow_inspection_is_unknown_not_zero(self):
        _, episode, statuses = analyze(NESTED_RENAMED, PolicyLimits(mapping_depth=0))
        self.assertEqual(statuses[0], UNREADABLE)
        self.assert_withheld(episode)

    def test_a_search_outside_the_grammar_is_unknown_not_zero(self):
        """Breadth-first search without a cost map: no binding, but a worklist."""
        bank, episode, statuses = analyze(BFS_WITHOUT_COSTS)
        self.assertEqual(statuses[0], UNREADABLE)
        reasons = {a.get("unreadable") for a in bank.audit["parts"].values()}
        self.assertIn("worklist_without_binding", reasons)
        self.assert_withheld(episode)

    def test_ordinary_loops_and_reactive_policies_still_count_as_zero(self):
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
        bank, episode, statuses = analyze(
            source, development=((1, 0),), validation=((3, 1),)
        )
        self.assertEqual(statuses, ["not_search_shaped"])
        self.assertTrue(any(r["part_status"] == UNREADABLE for r in episode["runs"]))
        self.assertFalse(episode["accounting"]["program"]["complete"])


class WiderFrontierTest(unittest.TestCase):
    """Common ways of writing the same searches are read, not left unknown."""

    def test_tie_break_numbers_in_heap_entries_are_read(self):
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
        run = run_of({"q": [1, 2]}, {"q": [2, 3, 4]})
        self.assertEqual(worklist_signature(run)["worklist"], "q")

    def test_only_popping_or_only_appending_is_not_a_worklist(self):
        for frames in (
            ({"q": [1, 2, 3]}, {"q": [2, 3]}, {"q": [3]}),
            ({"p": [1]}, {"p": [1, 2]}),
        ):
            self.assertIsNone(worklist_signature(run_of(*frames))["worklist"])

    def test_deques_sets_and_nested_containers_are_seen(self):
        run = run_of({"box": {"q": deque([1])}}, {"box": {"q": deque([2])}})
        self.assertEqual(worklist_signature(run)["worklist"], 'box["q"]')
        found, cut = containers({"s": {1, 2}, "deep": [[[[[1]]]]]}, depth=2)
        self.assertIn("s", found)
        self.assertTrue(cut)


class DequeAndTaggedEditTest(unittest.TestCase):
    """Live edits on deques and on entries with tie-break numbers."""

    def test_edits_address_entries_by_priority_and_cell(self):
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
        run = run_of({"q": deque([(1, 2)], 5), "seen": {(1, 2)}})
        _, values = next(run.capture.frames())
        self.assertEqual(values["q"], deque([(1, 2)], 5))
        self.assertEqual(values["q"].maxlen, 5)
        self.assertEqual(values["seen"], {(1, 2)})


if __name__ == "__main__":
    unittest.main()
