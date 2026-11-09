"""Behavioral and adversarial checks for the separate search-explanation family."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest import mock

from robocode.planning_explanations.instrument import UnsupportedProgram
from robocode.planning_explanations.search.__main__ import main
from robocode.planning_explanations.search.checkpoints import Edit, Program
from robocode.planning_explanations.search.discover import (
    Limits,
    discover,
    load_frozen,
    validate,
)
from robocode.planning_explanations.search.engine import (
    Action,
    SearchSpec,
    initial,
    run,
    step,
)
from robocode.planning_explanations.search.environment import check_environment, replay
from robocode.planning_explanations.search.explanation import Explanation, Mapping
from robocode.planning_explanations.search.fixtures import FIXTURES
from robocode.planning_explanations.search.report import paired_svg
from robocode.planning_explanations.search.verify import compare, intervene, verify


def graph(selection="fifo"):
    """A supplied milestone-2 reference; discovery tests do not receive it."""
    return Explanation(
        SearchSpec(selection=selection), Mapping("5:4:While", "root", "todo", "parent")
    )


class EngineTests(unittest.TestCase):
    """Independent Q semantics, without concrete programs or a discovery oracle."""

    def test_fifo_and_lifo_expansion_counts(self):
        """Goal selection is not expansion; queue discipline changes the work."""
        for selection, expected in (("fifo", [2, 2, 0]), ("lifo", [2, 1, 0])):
            for root, count in enumerate(expected):
                execution = run(SearchSpec(selection=selection), root)
                expanded = {
                    t.expanded for t in execution.transitions if t.expanded is not None
                }
                self.assertEqual(len(expanded), count)
                self.assertEqual(
                    execution.transitions[-1].state.output, 0 if root == 2 else 1
                )
                self.assertEqual(execution.transitions[-1].operation, "goal")

    def test_first_parents_and_immutable_step_inputs(self):
        """Repeated neighbors keep their first parent and never re-enter frontier."""
        original = initial(0)
        one = step(SearchSpec(), original).state
        two = step(SearchSpec(), one)
        self.assertEqual(original, initial(0))
        self.assertEqual(one.parents, {0: None, 1: 0})
        self.assertEqual(two.generated, (0, 2))
        self.assertEqual(two.added, (2,))
        self.assertEqual(two.state.parents, {0: None, 1: 0, 2: 1})

    def test_action_child_baseline_is_one_expansion(self):
        """Terminal action leaves are not additional environment-node expansions."""
        for root in range(3):
            execution = run(SearchSpec(mode="reactive"), root)
            self.assertEqual(len(execution.transitions), 2)
            self.assertEqual(execution.transitions[0].expanded, root)
            self.assertIsInstance(execution.transitions[1].selected, Action)
            self.assertIsNone(execution.transitions[1].expanded)

    def test_fallback_and_budget(self):
        """An empty frontier uses the declared fallback; exhaustion is explicit."""
        cleared = intervene(initial(0), "clear")
        self.assertEqual(step(SearchSpec(), cleared).state.output, 0)
        with self.assertRaises(RuntimeError):
            run(SearchSpec(), 0, max_steps=1)
        with self.assertRaises(ValueError):
            intervene(initial(0), "drop", 2)

    def test_environment_meaning_and_witness_replay(self):
        """Weak action witnesses are replayed and never charged to the policy."""
        evidence = check_environment(SearchSpec())
        self.assertEqual(evidence["status"], "supported")
        self.assertEqual(evidence["attributed_policy_work"], 0)
        self.assertEqual(len(evidence["goal_checks"]), 6)
        for witness in evidence["witnesses"]:
            self.assertEqual(
                replay(witness["start"], tuple(witness["actions"])),
                witness["trajectory"],
            )
        self.assertEqual(
            check_environment(SearchSpec(block=3, goal=1))["reason"], "goal_meaning"
        )
        self.assertEqual(
            check_environment(SearchSpec(), max_length=0)["status"], "unresolved"
        )


class CheckpointTests(unittest.TestCase):
    """Observation, live edits, current-state mapping and negative causal controls."""

    def test_plain_observation_parity_all_fixture_inputs(self):
        """All six states agree with the separately compiled original function."""
        for name, source in FIXTURES.items():
            program = Program(source)
            for position in range(6):
                with self.subTest(name=name, position=position):
                    trace = program.observe(position)
                    self.assertEqual(trace.output, 1 if position < 4 else 0)
                    self.assertEqual(trace.edits, 0)
                    self.assertEqual(trace.snapshots[0].kind, "entry")
                    self.assertEqual(trace.snapshots[-1].kind, "exit")

    def test_live_list_edits_preserve_aliases(self):
        """Editing the selected frontier mutates other aliases in real execution."""
        program = Program("def f(p):\n    a = [1, 2]\n    b = a\n    return len(b)\n")
        for operation, value, output, contents in (
            ("clear", 0, 0, []),
            ("drop", 0, 1, [2]),
            ("reverse", 0, 2, [2, 1]),
        ):
            patched = program.run(0, Edit("4:4:Return", 0, "a", operation, value))
            snapshot = next(s for s in patched.snapshots if s.before is not None)
            self.assertEqual(patched.output, output)
            self.assertEqual(snapshot.values["b"], contents)
            self.assertIs(snapshot.values["a"], snapshot.values["b"])
            self.assertIn(("a", "b"), snapshot.aliases)

    def test_scalar_edit_affects_subsequent_code_but_exit_cannot_be_patched(self):
        """The injection changes a local binding before use, never the output hook."""
        program = Program(FIXTURES["reactive"])
        self.assertEqual(
            program.run(0, Edit("3:4:Return", 0, "action", "set", -1)).output, -1
        )
        with self.assertRaises(ValueError):
            program.run(0, Edit("exit:3:4:Return", 0, "action", "set", -1))

    def test_unsupported_and_nonterminating_programs(self):
        """Unsupported representations and execution budgets stay explicit."""
        with self.assertRaises(UnsupportedProgram):
            Program("import os\ndef f(p):\n    return 0\n")
        with self.assertRaises(RuntimeError):
            Program("def f(p):\n    while True:\n        p = p\n").run(0, budget=100)
        with self.assertRaises(UnsupportedProgram):
            Program("def f(p):\n    a = []\n    a.append(a)\n    return 0\n").run(0)

    def test_configuration_mapping_is_not_output_matching(self):
        """FIFO and LIFO have the same decisions but disagree on intermediate work."""
        program = Program(FIXTURES["fifo"])
        result = compare(program, graph("lifo"), program.observe(2))
        self.assertEqual(result["predicted_output"], result["observed_output"])
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "configuration_mismatch")

    def test_decoy_passes_ordinary_but_fails_intervention(self):
        """Unused search is not supported by agreement of ordinary configurations."""
        program = Program(FIXTURES["decoy"])
        for position in (0, 2, 4):
            self.assertEqual(
                compare(program, graph(), program.observe(position))["status"],
                "supported",
            )
        result = verify(program, graph(), (0, 2, 4))
        counterexample = result["counterexample"]
        self.assertEqual(result["reason"], "intervention")
        self.assertTrue(counterexample["prefix_preserved"])
        self.assertEqual(counterexample["predicted_output"], 0)
        self.assertEqual(counterexample["observed_output"], 1)
        self.assertTrue(
            counterexample["comparison"]["alignments"][0]["edit_correspondence"]
        )

    def test_edit_correspondence_checked_before_continuation(self):
        """An incorrect concrete lift fails even when it preserves replay prefix."""
        program = Program(FIXTURES["fifo"])
        original = program.run

        def wrong_lift(position, edit=None):
            return original(
                position, replace(edit, operation="reverse") if edit else None
            )

        with mock.patch.object(program, "run", side_effect=wrong_lift):
            result = verify(program, graph(), (0,))
        counterexample = result["counterexample"]
        self.assertEqual(result["status"], "rejected")
        self.assertTrue(counterexample["prefix_preserved"])
        self.assertFalse(
            counterexample["comparison"]["alignments"][0]["edit_correspondence"]
        )

    def test_supplied_explanation_and_ordinary_only_evidence(self):
        """The verifier works independently; ordinary evidence is insufficient."""
        program = Program(FIXTURES["fifo"])
        self.assertEqual(verify(program, graph(), (0, 2, 4))["status"], "supported")
        result = verify(program, graph(), (0, 2, 4), interventions=False)
        self.assertEqual(
            (result["status"], result["reason"]), ("unresolved", "ordinary_only")
        )
        self.assertEqual(verify(program, graph(), ())["reason"], "no_inputs")

    def test_overwritten_action_is_not_a_causal_mapping(self):
        """An earlier copy of a decision can agree but not mediate the output."""
        program = Program(
            "def f(p):\n    a = 1 if p < 4 else 0\n"
            "    a = 1 if p < 4 else 0\n    return a\n"
        )
        candidate = Explanation(
            SearchSpec(mode="reactive"), Mapping("3:4:Assign", action="a")
        )
        self.assertEqual(
            compare(program, candidate, program.observe(0))["status"], "supported"
        )
        self.assertEqual(verify(program, candidate, (0,))["reason"], "intervention")


class DiscoveryTests(unittest.TestCase):
    """Automatic bindings, frozen selection, fresh failures, and exported evidence."""

    @classmethod
    def setUpClass(cls):
        cls.programs = {name: Program(source) for name, source in FIXTURES.items()}
        cls.banks = {name: discover(program) for name, program in cls.programs.items()}

    def test_discovers_graphs_and_retains_ambiguity(self):
        """Renaming/relabeling does not require semantic local identifiers."""
        for name in ("fifo", "lifo", "renamed"):
            bank = self.banks[name]
            selected = next(
                h for h in bank.explanations if h.identifier == bank.selected
            )
            self.assertEqual(selected.spec.mode, "graph")
            self.assertGreater(len(bank.explanations), 1)
            self.assertEqual(
                selected.spec.selection, "lifo" if name == "lifo" else "fifo"
            )
            if name == "renamed":
                self.assertEqual(selected.mapping.offset, 10)
                self.assertEqual(selected.mapping.frontier, "b")
            self.assertFalse(bank.audit["truncated"])

    def test_negative_controls_keep_minimal_explanations(self):
        """Baseline support is retained separately from rejection of rich search."""
        for name in ("reactive", "lookup", "decoy"):
            bank = self.banks[name]
            self.assertTrue(bank.explanations)
            self.assertTrue(all(h.spec.mode == "reactive" for h in bank.explanations))
            report = validate(self.programs[name], bank)
            self.assertTrue(
                all(r["ordinary_mean_M"] == 0 for r in report["explanations"])
            )
            self.assertIsNone(report["planningness"])
        decoy = self.banks["decoy"]
        self.assertTrue(
            any(r["reason"] == "intervention" for r in decoy.audit["rejected"])
        )

    def test_bounded_search_is_not_a_zero_planning_conclusion(self):
        """A deliberately insufficient candidate budget reports truncation."""
        bank = discover(self.programs["fifo"], limits=Limits(candidates=1))
        self.assertTrue(bank.audit["truncated"])
        self.assertIsNone(bank.selected)
        result = validate(self.programs["fifo"], bank)
        self.assertEqual(result["explanations"], [])
        self.assertIsNone(result["planningness"])

    def test_fresh_mean_uses_only_ordinary_runs(self):
        """Patch counts never enter the specified state distribution."""
        report = validate(
            self.programs["fifo"],
            self.banks["fifo"],
            weights={1: 0.25, 3: 0.25, 5: 0.5},
        )
        selected = next(
            r for r in report["explanations"] if r["id"] == report["selected"]
        )
        self.assertEqual(selected["ordinary_mean_N"], 1)
        self.assertEqual(selected["ordinary_mean_M"], 0.5)
        self.assertEqual([r["N"] for r in selected["ordinary"]], [2, 2, 0])
        self.assertGreater(len(selected["interventions"]), len(selected["ordinary"]))
        self.assertTrue(
            all("N" not in r["comparison"] for r in selected["interventions"])
        )

    def test_frozen_roundtrip_no_search_during_validation(self):
        """Reloaded hypotheses retain identity and validation cannot propose new ones."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frozen.json"
            self.banks["fifo"].save(path)
            loaded = load_frozen(path)
            self.assertEqual(loaded.fingerprint, self.banks["fifo"].fingerprint)
            with mock.patch(
                "robocode.planning_explanations.search.discover.specifications",
                side_effect=AssertionError("No discovery"),
            ):
                result = validate(self.programs["fifo"], loaded)
            self.assertEqual(result["selected"], loaded.selected)
            data = json.loads(path.read_text())
            data["selected"] = None
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_frozen(path)

    def test_source_and_input_separation(self):
        """The bank is tied to source bytes and disjoint input partitions."""
        with self.assertRaises(ValueError):
            validate(self.programs["lifo"], self.banks["fifo"])
        with self.assertRaises(ValueError):
            validate(self.programs["fifo"], self.banks["fifo"], (0, 1))
        with self.assertRaises(ValueError):
            validate(
                self.programs["fifo"],
                self.banks["fifo"],
                weights={1: 0.5, 3: 0.5, 5: 0.5},
            )

    def test_heldout_counterexample_does_not_trigger_reselection(self):
        """A discovered explanation can fail fresh validation and stays failed."""
        source = FIXTURES["fifo"].replace(
            "return action", "return 0 if pos == 3 else action"
        )
        program = Program(source)
        bank = discover(program)
        self.assertTrue(bank.explanations)
        result = validate(program, bank)
        self.assertEqual(result["selected"], bank.selected)
        selected = next(r for r in result["explanations"] if r["id"] == bank.selected)
        self.assertEqual(selected["status"], "rejected")
        self.assertIsNone(selected["ordinary_mean_N"])
        self.assertIsNone(selected["ordinary_mean_M"])

    def test_cli_exports_every_intervention_and_reloadable_bank(self):
        """The demonstration is runnable and its machine/visual artifacts agree."""
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            self.assertEqual(main(["--output-dir", temporary]), 0)
            summaries = json.loads((Path(temporary) / "summary.json").read_text())
            self.assertEqual(len(summaries), len(FIXTURES))
            directory = Path(temporary) / "fifo"
            report = json.loads((directory / "report.json").read_text())
            page = (directory / "report.html").read_text()
            for result in report["validation"]["explanations"]:
                count = len(result["interventions"])
                self.assertIn(f"Intervention {count}/{count}", page)
            self.assertTrue((directory / "ordinary-traces.json").is_file())
            self.assertEqual(
                load_frozen(directory / "frozen.json").selected,
                self.banks["fifo"].selected,
            )
            ET.fromstring((directory / "ordinary.svg").read_text())
            ET.fromstring((directory / "intervention.svg").read_text())
            self.assertEqual(
                main(
                    [
                        "--fixture",
                        "fifo",
                        "--frozen",
                        str(directory / "frozen.json"),
                        "--output-dir",
                        str(Path(temporary) / "replay"),
                    ]
                ),
                0,
            )

    def test_svg_keeps_prediction_and_observation_distinct(self):
        """A failed comparison must display actual observed values, not copy Q."""
        result = verify(self.programs["decoy"], graph(), (0,))
        svg = paired_svg(result["counterexample"]["comparison"], "Decoy rejection")
        self.assertIn("output=0", svg)
        self.assertIn("output=1", svg)
        self.assertIn("FAIL", svg)
        ET.fromstring(svg)


if __name__ == "__main__":
    unittest.main()
