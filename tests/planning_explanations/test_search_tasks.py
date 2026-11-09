"""Milestones 4–5: findings, task shifts, causal controls and frozen experiments."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest import mock

from robocode.planning_explanations.search.__main__ import main
from robocode.planning_explanations.search.benchmark import (
    assess_task,
    check_partition,
    discover_tasks,
    load_task_bank,
    validate_tasks,
)
from robocode.planning_explanations.search.checkpoints import Program
from robocode.planning_explanations.search.discover import Limits, discover, validate
from robocode.planning_explanations.search.engine import SearchSpec, run
from robocode.planning_explanations.search.evidence import (
    computation_features,
    summarize_findings,
)
from robocode.planning_explanations.search.explanation import Explanation, Mapping
from robocode.planning_explanations.search.fixtures import FIXTURES
from robocode.planning_explanations.search.task_fixtures import (
    ADAPTIVE,
    GROUND_TRUTH,
    LINE,
    REVERSED,
    TASK_FIXTURES,
    benchmark_cases,
)
from robocode.planning_explanations.search.tasks import (
    TaskCase,
    TaskSearchSpec,
    task_environment,
)
from robocode.planning_explanations.search.verify import verify


class FindingTests(unittest.TestCase):
    """Search evidence, task success and N/M must not be interchangeable."""

    def test_one_expansion_is_search_even_when_m_is_zero(self):
        """The already-existing LIFO case is a positive M=0 control."""
        program = Program(FIXTURES["lifo"])
        explanation = Explanation(
            SearchSpec(selection="lifo"), Mapping("5:4:While", "root", "todo", "parent")
        )
        evidence = verify(program, explanation, (3,))
        row = evidence["ordinary"][0]
        self.assertEqual((row["N"], row["M"]), (1, 0))
        self.assertEqual(row["search_expansions"], 1)
        self.assertGreater(row["decision_effects"], 0)
        self.assertEqual(summarize_findings([evidence])["code"], "search_supported")

    def test_reactive_root_expansion_does_not_count_future_states(self):
        """Action children belong to the baseline, not hypothetical world states."""
        execution = run(SearchSpec(mode="reactive"), 0)
        features = computation_features(execution.transitions)
        self.assertEqual((features["N"], features["M"]), (1, 0))
        self.assertEqual(features["generated_alternatives"], 0)
        self.assertEqual(features["search_expansions"], 0)
        self.assertEqual(features["max_discovery_depth"], 0)

    def test_depth_alternatives_and_expansions_are_distinct(self):
        """A reconsidered parent counts as a generated alternative, not a new node."""
        features = computation_features(run(SearchSpec(), 0).transitions)
        self.assertEqual(features["N"], 2)
        self.assertEqual(features["max_discovery_depth"], 2)
        self.assertEqual(features["max_expanded_depth"], 1)
        self.assertEqual(features["generated_alternatives"], 3)
        self.assertEqual(features["max_frontier_size"], 1)

    def test_budget_failure_is_unresolved_not_reactive(self):
        """No supported explanation is different from supported reactive behavior."""
        program = Program(FIXTURES["fifo"])
        bank = discover(program, limits=Limits(candidates=1))
        self.assertEqual(validate(program, bank)["finding"]["code"], "unresolved")
        for name in ("reactive", "lookup"):
            program = Program(FIXTURES[name])
            self.assertEqual(
                validate(program, discover(program))["finding"]["code"], "reactive_only"
            )

    def test_ordinary_only_cannot_get_search_finding(self):
        """Matching configurations alone do not receive the causal-support label."""
        program = Program(FIXTURES["fifo"])
        hypothesis = Explanation(
            SearchSpec(), Mapping("5:4:While", "root", "todo", "parent")
        )
        result = verify(program, hypothesis, (0,), interventions=False)
        self.assertEqual(summarize_findings([result])["code"], "unresolved")

    def test_intervention_budget_failure_remains_unresolved(self):
        """A replay timeout is missing evidence, not a counterexample to planning."""
        program = Program(FIXTURES["fifo"])
        hypothesis = Explanation(
            SearchSpec(), Mapping("5:4:While", "root", "todo", "parent")
        )
        original = program.run

        def budgeted(position, edit=None):
            if edit:
                raise RuntimeError("Concrete line budget exhausted")
            return original(position)

        with mock.patch.object(program, "run", side_effect=budgeted):
            result = verify(program, hypothesis, (0,))
        self.assertEqual(result["mechanism_status"], "unresolved")
        self.assertEqual(summarize_findings([result])["code"], "unresolved")

    def test_unsupported_source_exports_explicit_unknown(self):
        """Unsupported syntax receives no numerical or binary planning conclusion."""
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            source = Path(temporary) / "lookup.py"
            source.write_text("import os\ndef f(p):\n    return 0\n")
            output = Path(temporary) / "reports"
            self.assertEqual(
                main(["--source", str(source), "--output-dir", str(output)]), 1
            )
            report = json.loads((output / "lookup" / "report.json").read_text())
            self.assertEqual(report["finding"]["code"], "unresolved")
            self.assertIsNone(report["fixture_ground_truth"])
            self.assertIsNone(report["ordinary_mean_M"])


class TaskBenchmarkTests(unittest.TestCase):
    """A joint frozen bank is evaluated on changed maps and goals, not refitted."""

    @classmethod
    def setUpClass(cls):
        cls.programs = {name: Program(source) for name, source in TASK_FIXTURES.items()}
        cls.banks = {
            name: discover_tasks(program, *benchmark_cases(name))
            for name, program in cls.programs.items()
        }
        cls.results = {
            name: validate_tasks(program, cls.banks[name])
            for name, program in cls.programs.items()
        }

    def test_native_and_instrumented_outputs_agree(self):
        """A fresh input dictionary is supplied to both executions and every replay."""
        for name, program in self.programs.items():
            development, validation_cases = benchmark_cases(name)
            for case in (*development, *validation_cases):
                with self.subTest(name=name, case=case.name):
                    self.assertEqual(
                        program.observe(case).output, program.original(case)
                    )
        program = Program(
            'def f(c):\n    c["transitions"][0][1] = None\n    return 0\n'
        )
        case = TaskCase("mutation", 0, 4)
        program.observe(case)
        self.assertEqual(case.transitions, LINE)

    def test_development_outputs_agree_for_positive_and_negative_controls(self):
        """The benchmark cannot be solved by development I/O agreement alone."""
        for case in benchmark_cases("adaptive")[0]:
            actions = {
                self.programs[name].original(case)
                for name in ("adaptive", "reactive", "lookup", "wrong_model", "decoy")
            }
            self.assertEqual(len(actions), 1)

    def test_heldout_cases_change_maps_and_goals(self):
        """State-input novelty alone is not the validation criterion anymore."""
        development, validation_cases = benchmark_cases("adaptive")
        self.assertTrue(
            any(
                c.transitions not in {d.transitions for d in development}
                for c in validation_cases
            )
        )
        self.assertTrue(
            any(c.goal not in {d.goal for d in development} for c in validation_cases)
        )
        self.assertTrue(
            any(c.start in {d.start for d in development} for c in validation_cases)
        )
        check_partition(development, validation_cases)
        with self.assertRaises(ValueError):
            check_partition(
                development, (replace(development[0], name="cosmetically_new"),)
            )
        with self.assertRaises(ValueError):
            check_partition(
                development, (replace(development[0], name="new_start", start=1),)
            )

    def test_adaptive_search_generalizes_with_frozen_input_ingredients(self):
        """Equivalent local/input root selections survive changed dynamics and goals."""
        result = self.results["adaptive"]
        self.assertEqual(result["finding"]["code"], "search_supported")
        supported = [r for r in result["explanations"] if r["status"] == "supported"]
        self.assertEqual(len(supported), 2)
        self.assertEqual(
            supported[0]["mechanism_mean_features"],
            supported[1]["mechanism_mean_features"],
        )
        hypothesis = next(
            h
            for h in self.banks["adaptive"].explanations
            if h.identifier == supported[0]["id"]
        )
        self.assertEqual(
            (hypothesis.spec.model_source, hypothesis.spec.goal_source),
            ("input", "input"),
        )
        self.assertEqual(result["task_performance"]["success_rate"], 1)
        self.assertEqual(
            [r["observed_output"] for r in supported[0]["ordinary"]], [-1, -1, -1]
        )

    def test_command_labels_are_not_node_differences(self):
        """Reversed edges and shortcuts require decoding the actual action label."""
        spec = TaskSearchSpec().bind(TaskCase("reversed", 0, 4, REVERSED))
        self.assertEqual(run(spec, 0).transitions[-1].state.output, -1)
        shortcut = benchmark_cases("adaptive")[1][-1]
        self.assertEqual(
            run(TaskSearchSpec().bind(shortcut), 0).transitions[-1].state.output, -1
        )

    def test_wrong_model_has_causal_search_but_fails_grounding(self):
        """Failed task semantics do not erase independently verified search work."""
        result = self.results["wrong_model"]
        self.assertEqual(result["finding"]["code"], "search_mechanism_only")
        evidence = next(
            r
            for r in result["explanations"]
            if r["mode"] == "graph" and r["mechanism_status"] == "supported"
        )
        self.assertEqual(evidence["status"], "rejected")
        self.assertEqual(evidence["environment"]["status"], "rejected")
        self.assertIsNone(evidence["ordinary_mean_N"])
        self.assertGreater(evidence["mechanism_mean_features"]["N"], 0)
        self.assertTrue(all(r["passed"] for r in evidence["interventions"]))
        self.assertLess(result["task_performance"]["success_rate"], 1)
        self.assertEqual(
            evidence["environment"]["cases"][0]["counterexamples"][0]["kind"], "edge"
        )

    def test_environment_checker_catches_unvisited_wrong_edge(self):
        """The identity-state task check covers all labeled edges, not only a path."""
        rows = list(LINE)
        rows[5] = (0, None)
        case = TaskCase("hidden_edge", 0, 1, tuple(rows))
        evidence = task_environment(TaskSearchSpec(model_source="fixed"), case)
        self.assertEqual(evidence["status"], "rejected")
        self.assertEqual(evidence["counterexamples"][0]["state"], 5)

    def test_lookup_and_decoy_retain_only_reactive_support(self):
        """Success on some tasks and unused search do not become planning evidence."""
        for name in ("reactive", "lookup", "decoy"):
            result = self.results[name]
            self.assertEqual(result["finding"]["code"], "reactive_only")
            self.assertTrue(
                all(
                    r["environment"]["status"] == "not_applicable"
                    for r in result["explanations"]
                    if r["mode"] == "reactive"
                )
            )
            self.assertTrue(
                all(
                    r["mode"] == "reactive"
                    for r in result["explanations"]
                    if r["status"] == "supported"
                )
            )
        self.assertTrue(
            any(
                r["reason"] == "intervention"
                for r in self.banks["decoy"].audit["rejected"]
            )
        )
        self.assertGreater(
            self.results["lookup"]["task_performance"]["success_rate"], 0
        )
        self.assertIsNone(self.results["lookup"]["planningness"])

    def test_single_expansion_fixture_has_search_support_and_zero_m(self):
        """One expansion across changed tasks is still genuine graph-search evidence."""
        result = self.results["one_expansion"]
        self.assertEqual(result["finding"]["code"], "search_supported")
        supported = next(
            r for r in result["explanations"] if r["status"] == "supported"
        )
        self.assertEqual(
            (supported["ordinary_mean_N"], supported["ordinary_mean_M"]), (1, 0)
        )
        self.assertTrue(all(r["search_expansions"] == 1 for r in supported["ordinary"]))

    def test_new_case_does_not_fit_or_replace_failed_selection(self):
        """Tied development explanations can fail later; no post-hoc repair occurs."""
        bank = self.banks["wrong_model"]
        with mock.patch(
            "robocode.planning_explanations.search.benchmark.task_specifications",
            side_effect=AssertionError("Cannot propose"),
        ):
            result = validate_tasks(self.programs["wrong_model"], bank)
        self.assertEqual(result["selected"], bank.selected)
        selected = next(r for r in result["explanations"] if r["id"] == bank.selected)
        self.assertEqual(selected["status"], "rejected")
        self.assertIsNone(selected["ordinary_mean_M"])

    def test_discovery_does_not_execute_validation_cases(self):
        """Held-out declarations may be frozen, but are not run during proposals."""
        program = Program(TASK_FIXTURES["lookup"])
        development, validation_cases = benchmark_cases("lookup")
        original = program.run

        def guarded(case, edit=None):
            self.assertIn(case, development)
            return original(case, edit)

        with mock.patch.object(program, "run", side_effect=guarded):
            discover_tasks(program, development, validation_cases)

    def test_frozen_roundtrip_source_and_case_integrity(self):
        """Case contents, their weights, hypotheses and source are frozen together."""
        bank = self.banks["adaptive"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frozen.json"
            bank.save(path)
            loaded = load_task_bank(path)
            self.assertEqual(loaded.fingerprint, bank.fingerprint)
            self.assertEqual(
                validate_tasks(self.programs["adaptive"], loaded)["finding"]["code"],
                "search_supported",
            )
            with self.assertRaises(ValueError):
                validate_tasks(self.programs["lookup"], loaded)
            data = json.loads(path.read_text())
            data["validation_cases"][0]["goal"] = 1
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_task_bank(path)

    def test_rollouts_and_interventions_do_not_enter_feature_average(self):
        """Population averages contain one ordinary invocation per frozen case."""
        result = self.results["adaptive"]
        evidence = next(r for r in result["explanations"] if r["status"] == "supported")
        mean = sum(r["N"] for r in evidence["ordinary"]) / len(evidence["ordinary"])
        self.assertAlmostEqual(evidence["ordinary_mean_N"], mean)
        self.assertTrue(
            all("N" not in p["comparison"] for p in evidence["interventions"])
        )
        self.assertTrue(
            all(
                r["attributed_policy_work"] == 0
                for r in result["task_performance"]["cases"]
            )
        )

    def test_rollout_outcomes_are_separate_from_mechanism(self):
        """Success, invalid actions and deterministic cycles have inspectable records."""
        case = TaskCase("simple", 0, 1)
        self.assertTrue(assess_task(self.programs["lookup"], case)["goal_reached"])
        failed = assess_task(Program("def f(c):\n    return 0\n"), case)
        self.assertEqual(failed["reason"], "deterministic_cycle")
        failed = assess_task(Program("def f(c):\n    return -1\n"), case)
        self.assertEqual(failed["status"], "failed")

    def test_renaming_bindings_preserves_task_discovery(self):
        """No benchmark fixture names or local role names reach the proposer."""
        source = (
            ADAPTIVE.replace("root", "r17")
            .replace("todo", "z42")
            .replace("parent", "p9")
            .replace("policy", "f")
        )
        program = Program(source)
        bank = discover_tasks(program, *benchmark_cases("adaptive"))
        result = validate_tasks(program, bank)
        self.assertEqual(result["finding"]["code"], "search_supported")
        self.assertTrue(
            any(
                h.mapping.frontier == "z42"
                for h in bank.explanations
                if h.spec.mode == "graph"
            )
        )

    def test_candidate_budget_is_explicit(self):
        """Truncating enumeration cannot produce a negative planning conclusion."""
        bank = discover_tasks(
            self.programs["adaptive"],
            *benchmark_cases("adaptive"),
            limits=Limits(candidates=1),
        )
        self.assertTrue(bank.audit["truncated"])
        self.assertEqual(
            validate_tasks(self.programs["adaptive"], bank)["finding"]["code"],
            "unresolved",
        )

    def test_cli_exports_findings_features_and_separate_truth(self):
        """Reports retain separate evidence axes and every validation intervention."""
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            self.assertEqual(main(["--suite", "tasks", "--output-dir", temporary]), 0)
            summaries = json.loads((Path(temporary) / "summary.json").read_text())
            self.assertEqual(len(summaries), len(TASK_FIXTURES))
            for summary in summaries:
                directory = Path(temporary) / summary["fixture"]
                report = json.loads((directory / "report.json").read_text())
                self.assertEqual(
                    report["fixture_ground_truth"]["label"],
                    GROUND_TRUTH[summary["fixture"]],
                )
                self.assertEqual(report["validation"]["finding"], summary["finding"])
                page = (directory / "report.html").read_text()
                self.assertIn("Separate task success", page)
                self.assertIn("Mechanism:", page)
                for result in report["validation"]["explanations"]:
                    n = len(result["interventions"])
                    if n:
                        self.assertIn(f"Intervention {n}/{n}", page)
                ET.fromstring((directory / "ordinary.svg").read_text())
            directory = Path(temporary) / "adaptive"
            self.assertEqual(
                main(
                    [
                        "--suite",
                        "tasks",
                        "--fixture",
                        "adaptive",
                        "--frozen",
                        str(directory / "frozen.json"),
                        "--output-dir",
                        str(Path(temporary) / "replay"),
                    ]
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
