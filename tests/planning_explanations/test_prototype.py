"""Behavioral tests for the bounded planning-explanation experiment.

Run with unittest (stdlib only), or collect these tests with the repo's pytest.
"""

import ast
import dataclasses
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from robocode.planning_explanations.__main__ import main
from robocode.planning_explanations.discovery import (
    Explanation,
    Limits,
    discover,
    ordinary,
    paired_test,
    validate,
)
from robocode.planning_explanations.fixtures import FIXTURES, ONE_STEP, TWO_STEP
from robocode.planning_explanations.instrument import (
    Patch,
    Program,
    Selector,
    UnsupportedProgram,
)
from robocode.planning_explanations.report import (
    frozen_payload,
    load_frozen,
    save_report,
    write_json,
)
from robocode.planning_explanations.task import (
    Navigation,
    cases,
    make_case,
    rollout,
)


class PrototypeTests(unittest.TestCase):
    """Exercise positive, negative, ambiguous and unresolved cases together."""

    @classmethod
    def setUpClass(cls):
        cls.training = cases(17, "discovery")
        cls.fresh = cases(29, "validation")
        cls.programs = {name: Program(source) for name, source in FIXTURES.items()}
        cls.banks = {
            name: discover(p, cls.training) for name, p in cls.programs.items()
        }
        cls.results = {
            name: validate(p, cls.banks[name], cls.fresh)
            for name, p in cls.programs.items()
        }

    def test_observation_preserves_all_fixture_outputs_and_inputs(self):
        """Trace every fixture without changing outputs or input mutation."""
        for name, program in self.programs.items():
            for case in self.training + self.fresh:
                with self.subTest(program=name, case=case.label):
                    self.assertEqual(program.check_observation(case).patch_count, 0)

    def test_observation_preserves_aliases_mutation_and_short_circuiting(self):
        """Hooks must preserve Python object and evaluation semantics."""
        program = Program("""\
def p(s, t, u, c):
    x = t[0]
    y = x
    y[0] = 5
    z = False and u[100]
    return x[0] + t[0][0]
""")
        trace = program.check_observation(self.training[0])
        self.assertEqual(trace.output, 10)
        self.assertEqual(trace.args_after[1][0][0], 5)
        self.assertEqual(self.training[0].transitions[0][0], 1)

    def test_nested_expression_checkpoints_are_distinct(self):
        """Nested subscripts sharing a start position need distinct sites."""
        program = self.programs["two_step"]
        sites = [
            site
            for site, info in program.sites.items()
            if info["expression"] in ("t[s]", "t[s][a]")
        ]
        self.assertEqual(len(sites), 2)
        self.assertNotEqual(sites[0], sites[1])

    def test_targeted_scalar_and_dictionary_interventions(self):
        """A dictionary leaf patch changes one run, preserving prior snapshots."""
        program = Program("""\
def p(s, t, u, c):
    d = {0: 3, 1: 7}
    return d[0] + d[1]
""")
        trace = program.run(self.training[0])
        event = next(e for e in trace.events if program.sites[e.site]["kind"] == "Dict")
        patched = program.run(
            self.training[0], Patch(Selector(event.site, 0, (0,)), 10)
        )
        self.assertEqual(trace.output, 10)
        self.assertEqual(patched.output, 17)
        self.assertEqual(patched.patch_count, 1)
        self.assertEqual(trace.event(Selector(event.site, 0)).value[0], 3)

    def test_recover_both_horizons_and_keep_ambiguity(self):
        """Recover horizon and alternatives without arbitrarily selecting one map."""
        for name, horizon in (
            ("one_step", 1),
            ("two_step", 2),
            ("renamed_dictionary", 2),
        ):
            with self.subTest(name=name):
                result = self.results[name]
                # One computation; its copies are interchangeable locations rather
                # than separate explanations.
                (explanation,) = self.banks[name].explanations
                self.assertTrue(explanation.interchangeable)
                for row in result["ordinary_population"]:
                    self.assertEqual(row["status"], "supported_planning")
                    self.assertEqual(row["features"]["supported_horizons"], [horizon])
                    self.assertEqual(
                        row["features"]["evaluated_alternatives"], [2**horizon]
                    )
                    self.assertTrue(row["ambiguity"])

    def test_arbitrary_identifier_renaming_does_not_change_recovery(self):
        """Remove all meaningful local names and recover the same horizon."""
        tree = ast.parse(TWO_STEP)
        names = sorted(
            {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            - {"range", "max", "int"}
        )
        names = {name: f"v{i}" for i, name in enumerate(names)}

        class Rename(ast.NodeTransformer):
            """Apply an arbitrary consistent renaming without adding role labels."""

            def visit_Name(self, node):  # pylint: disable=invalid-name
                """Rename identifier uses and assignments."""
                node.id = names.get(node.id, node.id)
                return node

            def visit_arg(self, node):
                """Rename input parameters consistently."""
                node.arg = names.get(node.arg, node.arg)
                return node

        tree = Rename().visit(tree)
        tree.body[0].name = "anonymous"
        program = Program(ast.unparse(tree))
        bank = discover(program, self.training)
        self.assertTrue(bank.explanations)
        self.assertEqual({e.horizon for e in bank.explanations}, {2})

    def test_rearrangement_recovers_reverse_action_enumeration(self):
        """Mappings must accommodate a different traversal order."""
        source = ONE_STEP.replace("scores = []", "scores = [0, 0]")
        source = source.replace("range(2)", "[1, 0]")
        source = source.replace("scores.append(value)", "scores[a] = value")
        program = Program(source)
        bank = discover(program, self.training)
        result = validate(program, bank, self.fresh)
        self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 1)

    def test_decoy_and_lookup_are_observationally_identical_to_planner(self):
        """Negative controls agree on outputs but receive no planning support."""
        for case in self.training + self.fresh:
            expected = self.programs["two_step"].run(case).output
            self.assertEqual(self.programs["decoy"].run(case).output, expected)
            self.assertEqual(self.programs["lookup"].run(case).output, expected)
        self.assertFalse(self.banks["decoy"].explanations)
        self.assertFalse(self.banks["lookup"].explanations)

    def test_same_planner_mapping_is_rejected_for_decoy_by_causal_effect(self):
        """Transfer the exact planner alignment and falsify its decision effect."""
        explanation = self.banks["two_step"].explanations[0]
        decoy = self.programs["decoy"]
        for case in self.training:
            self.assertIsNone(ordinary(explanation, case, decoy.run(case)))
        evidence, failure = paired_test(
            decoy, explanation, self.training[0], decoy.run(self.training[0])
        )
        self.assertEqual(failure["reason"], "intervention_mismatch")
        witness = evidence[-1]
        self.assertNotEqual(witness["predicted_output"], witness["observed_output"])
        self.assertEqual(witness["observed_output"], witness["baseline_output"])

    def test_internal_effects_are_checked_even_when_outputs_match(self):
        """Equal outputs cannot hide incorrect downstream state predictions."""
        program = self.programs["two_step"]
        supported = self.banks["two_step"].explanations[0]
        # An early row snapshot contains both states, but changing its second
        # element does not affect the later read used by the second branch.
        row = next(
            s for s, info in program.sites.items() if info["expression"] == "t[s]"
        )
        mapping = dict(supported.mapping)
        mapping["states_1"] = (Selector(row, 0, (0,)), Selector(row, 0, (1,)))
        explanation = Explanation(
            supported.route,
            2,
            tuple((role, mapping[role]) for role, _ in supported.mapping),
        )
        case = make_case("equal-utilities", 0, [0.25] * 6)
        evidence, failure = paired_test(program, explanation, case, program.run(case))
        self.assertEqual(failure["reason"], "intervention_mismatch")
        self.assertEqual(
            evidence[-1]["predicted_output"], evidence[-1]["observed_output"]
        )
        self.assertTrue(
            any(
                c["predicted"] != c["observed"]
                for c in evidence[-1]["downstream_checks"]
            )
        )

    def test_unrolled_code_is_supported(self):
        """Values read by separate expressions can still form one role."""
        for source, horizon in (
            ("def p(s, t, u, c):\n    return int(u[t[s][1]] > u[t[s][0]])\n", 1),
            (
                "def p(s, t, u, c):\n"
                "    left = t[s][0]\n"
                "    right = t[s][1]\n"
                "    a = max(u[t[left][0]], u[t[left][1]])\n"
                "    b = max(u[t[right][0]], u[t[right][1]])\n"
                "    return int(b > a)\n",
                2,
            ),
        ):
            with self.subTest(horizon=horizon):
                program = Program(source)
                bank = discover(program, self.training)
                result = validate(program, bank, self.fresh)
                self.assertEqual({e.horizon for e in bank.explanations}, {horizon})
                self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 1)

    def test_unused_copy_is_rejected_by_its_own_patch(self):
        """A read whose element the decision never uses cannot stand for a score."""
        program = self.programs["two_step"]
        # In `return int(scores[1] > scores[0])` the first read of `scores` only
        # feeds `scores[1]`, so its element 0 is never used by the decision.
        first_read = min(
            (
                site
                for site, info in program.sites.items()
                if info["expression"] == "scores" and info["line"] == 11
            ),
            key=lambda site: program.sites[site]["column"],
        )
        search = next(iter(self.banks["two_step"].audit["routes"].values()))[
            "horizons"
        ][1]
        rejected = [
            r["location"]
            for r in search["rejected_locations"]
            if (r["role"], r["coordinate"]) == ("scores", 0)
        ]
        self.assertIn(dataclasses.asdict(Selector(first_read, 0, (0,))), rejected)

    def test_search_cost_follows_locations_not_their_combinations(self):
        """Per-coordinate screening finds the mapping without enumerating products."""
        search = next(iter(self.banks["two_step"].audit["routes"].values()))[
            "horizons"
        ][1]
        self.assertEqual(search["result"], "supported")
        self.assertEqual(search["candidates_tested"], 1)
        # Matching keeps every copy: about four locations per coordinate, whose
        # product (millions) is never enumerated.
        self.assertGreater(sum(c["matched"] for c in search["coordinates"]), 40)

    def test_each_interchangeable_location_passes_the_full_test(self):
        """Pairwise compatibility must agree with the full paired test it replaces."""
        program = self.programs["two_step"]
        explanation = self.banks["two_step"].explanations[0]
        runs = [(c, program.run(c)) for c in self.training]
        swaps = list(explanation.swaps())
        self.assertTrue(swaps)
        for role, coordinate, location, variant in swaps:
            with self.subTest(role=role, coordinate=coordinate, location=location):
                for case, trace in runs:
                    self.assertIsNone(ordinary(variant, case, trace))
                for case, trace in runs[:2]:
                    self.assertIsNone(paired_test(program, variant, case, trace)[1])

    def test_conditional_routes_and_unknown_are_execution_specific(self):
        """Do not assign the planning route's features to reactive executions."""
        result = self.results["conditional"]
        routes = set()
        for row in result["ordinary_population"]:
            routes.add(row["route"])
            if row["state"] % 2:
                self.assertEqual(row["status"], "supported_planning")
                self.assertEqual(row["features"]["supported_horizons"], [2])
            else:
                self.assertEqual(row["status"], "unresolved")
                self.assertEqual(row["features"]["supported_horizons"], [])
                self.assertIsNone(row["features"]["amount_if_unresolved"])
        self.assertEqual(len(routes), 2)
        self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 0.5)
        self.assertIsNone(result["planningness"])

    def test_distribution_applies_only_to_ordinary_runs(self):
        """Changing state weights must not count intervention runs in the population."""
        weights = {0: 0.75, 1: 0.25, 2: 0, 3: 0, 4: 0, 5: 0}
        result = validate(
            self.programs["conditional"], self.banks["conditional"], self.fresh, weights
        )
        self.assertEqual(len(result["ordinary_population"]), len(self.fresh))
        self.assertAlmostEqual(
            sum(r["weight"] for r in result["ordinary_population"]), 1
        )
        self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 0.25)
        self.assertAlmostEqual(result["evidence_coverage"]["unresolved_mass"], 0.75)
        self.assertGreater(
            sum(len(e["interventions"]) for e in result["explanations"]),
            len(self.fresh),
        )

    def test_unseen_route_is_not_discovered_during_validation(self):
        """Validation must not fit an explanation for a newly encountered route."""
        program = self.programs["conditional"]
        bank = discover(program, [c for c in self.training if c.state % 2 == 0])
        before = bank.digest
        result = validate(program, bank, self.fresh)
        self.assertFalse(bank.explanations)
        self.assertEqual(bank.digest, before)
        self.assertTrue(
            all(r["status"] == "unresolved" for r in result["ordinary_population"])
        )

    def test_freeze_roundtrip_and_no_refit(self):
        """A serialized frozen bank validates unchanged in a separate load."""
        bank = self.banks["two_step"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            write_json(path, frozen_payload(bank))
            restored = load_frozen(path)
            before = restored.digest
            result = validate(self.programs["two_step"], restored, self.fresh)
            self.assertEqual(bank.explanations, restored.explanations)
            self.assertEqual(before, restored.digest)
            self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            bank.explanations[0].horizon = 1

    def test_validation_rejects_training_reuse_even_with_new_labels(self):
        """Check input contents and program identity before fresh validation."""
        copied = [dataclasses.replace(c, label="new-" + c.label) for c in self.training]
        with self.assertRaisesRegex(ValueError, "fresh"):
            validate(self.programs["two_step"], self.banks["two_step"], copied)
        with self.assertRaisesRegex(ValueError, "different program"):
            validate(self.programs["decoy"], self.banks["two_step"], self.fresh)

    def test_fresh_counterexample_rejects_frozen_explanation_without_repair(self):
        """A failure on new utility contexts invalidates the frozen candidate."""
        source = TWO_STEP.replace(
            "return int(scores[1] > scores[0])",
            "return int(scores[1] > scores[0]) + int(u[0] > 100)",
        )
        program = Program(source)
        bank = discover(program, self.training)
        self.assertTrue(bank.explanations)
        before = bank.digest
        fresh = [
            make_case(c.label, c.state, (101.25,) + c.utilities[1:]) for c in self.fresh
        ]
        result = validate(program, bank, fresh)
        self.assertEqual(bank.digest, before)
        self.assertTrue(all(e["status"] == "rejected" for e in result["explanations"]))
        self.assertTrue(all(e["counterexamples"] for e in result["explanations"]))
        self.assertAlmostEqual(result["evidence_coverage"]["unresolved_mass"], 1)

    def test_enumeration_limit_is_reported_as_unresolved_not_absence(self):
        """Budget exhaustion cannot silently become evidence of no planning."""
        bank = discover(self.programs["two_step"], self.training, Limits(locations=1))
        self.assertTrue(any(a["truncated"] for a in bank.audit["routes"].values()))
        result = validate(self.programs["two_step"], bank, self.fresh)
        self.assertIsNone(result["planningness"])
        with self.assertRaises(ValueError):
            Limits(intervention_cases=0)

    def test_invalid_distribution_and_nonfinite_utilities_rejected(self):
        """Reject undefined task utilities and invalid probability distributions."""
        for weights in ({s: 1 for s in range(6)}, {s: float("nan") for s in range(6)}):
            with self.assertRaises(ValueError):
                validate(
                    self.programs["two_step"],
                    self.banks["two_step"],
                    self.fresh,
                    weights,
                )
        with self.assertRaises(ValueError):
            make_case("invalid", 0, [float("inf")] * 6)

    def test_report_contains_reproducible_evidence_and_raw_traces(self):
        """Persist enough data to audit predictions, observations and populations."""
        with tempfile.TemporaryDirectory() as directory:
            report = save_report(
                directory,
                "two_step",
                self.programs["two_step"],
                self.banks["two_step"],
                self.fresh,
                self.results["two_step"],
            )
            payload = json.loads((Path(directory) / "report.json").read_text())
            self.assertEqual(payload["source_digest"], report["source_digest"])
            records = payload["validation"]["explanations"][0]["interventions"]
            self.assertTrue(records[0]["downstream_checks"])
            self.assertIn("predicted_roles", records[0])
            self.assertIn("observed_roles", records[0])
            self.assertIn("checkpoints", payload)
            self.assertEqual(
                len(json.loads((Path(directory) / "ordinary-traces.json").read_text())),
                len(self.fresh),
            )
            self.assertIn(
                "not a count of unique arithmetic operations",
                (Path(directory) / "report.md").read_text(),
            )

    def test_unsupported_python_fails_explicitly(self):
        """Reject syntax outside the documented trusted subset."""
        for source in (
            "import os",
            "def p(s,t,u,c):\n return sum(u)",
            "def p(s,t,u,c):\n return [x for x in u]",
        ):
            with self.assertRaises(UnsupportedProgram):
                Program(source)

    def test_depth_one_needs_no_second_location_for_the_same_number(self):
        """Depth 1 compares evaluations directly, so code style cannot block support.

        Appending ``u[...]`` directly leaves no later place holding each number on
        its own; a separate depth-1 score role used to demand one.
        """
        case = self.training[0]
        roles = rollout(case.state, case.transitions, case.utilities, 1)
        self.assertEqual(set(roles), {"states_1", "evaluations"})
        program = Program(
            "def p(s, t, u, c):\n"
            "    greedy = []\n"
            "    for a in range(2):\n"
            "        greedy.append(u[t[s][a]])\n"
            "    return int(greedy[1] > greedy[0])\n"
        )
        bank = discover(program, self.training)
        result = validate(program, bank, self.fresh)
        self.assertAlmostEqual(result["evidence_coverage"]["supported_mass"], 1)
        self.assertEqual({e.horizon for e in bank.explanations}, {1})

    def test_source_dir_analyzes_each_program_and_records_refusals(self):
        """A folder run reports supported programs and lists refused ones."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            programs = root / "programs"
            programs.mkdir()
            (programs / "planner.py").write_text(ONE_STEP)
            (programs / "loop.py").write_text(
                "def p(s, t, u, c):\n    while s:\n        s = 0\n    return 0\n"
            )
            (programs / "_skipped.py").write_text("not a policy")
            out = root / "out"
            argv = ["prog", "--source-dir", str(programs), "--output-dir", str(out)]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                main()  # type: ignore[no-untyped-call]
            rows = json.loads((out / "summary.json").read_text())
            summary = {row["program"]: row for row in rows}
            self.assertEqual(set(summary), {"loop", "planner"})
            self.assertIn("While", summary["loop"]["refused"])
            self.assertFalse((out / "loop").exists())
            self.assertGreater(summary["planner"]["validated_explanations"], 0)
            self.assertIn(
                "Refused: Unsupported syntax: While", (out / "README.md").read_text()
            )

    def test_environment_resets(self):
        """Environment trials start from the same supplied task state."""
        environment = Navigation(self.training[0])
        start = environment.reset()
        state = environment.step(1)
        self.assertEqual(state, self.training[0].transitions[start][1])
        self.assertEqual(
            environment.terminal_utility(), self.training[0].utilities[state]
        )
        self.assertEqual(environment.reset(), start)


if __name__ == "__main__":
    unittest.main()
