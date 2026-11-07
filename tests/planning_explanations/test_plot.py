"""Check that plots depict the saved evidence, not reconstructed rollouts."""

import argparse
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from importlib import import_module, util
from io import StringIO
from pathlib import Path

from robocode.planning_explanations.discovery import discover, validate
from robocode.planning_explanations.fixtures import CONDITIONAL, ONE_STEP
from robocode.planning_explanations.instrument import Program
from robocode.planning_explanations.report import save_report
from robocode.planning_explanations.task import cases

if util.find_spec("matplotlib") is None:
    raise unittest.SkipTest("Visualization tests require Matplotlib")

plot = import_module("robocode.planning_explanations.plot")


class PlotTests(unittest.TestCase):
    """Use fresh temporary reports; tests never depend on existing demo output."""

    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        for name, source in (("conditional", CONDITIONAL), ("one_step", ONE_STEP)):
            program = Program(source)
            bank = discover(program, cases(17, "discovery"))
            fresh = cases(29, "validation")
            save_report(
                cls.root / name,
                name,
                program,
                bank,
                fresh,
                validate(program, bank, fresh),
            )

    def render(self, fixture, case, directory, **options):
        """Run the same read-only plotting entry point as the CLI."""
        args = argparse.Namespace(
            report=self.root / fixture,
            case=case,
            explanation=None,
            list=False,
            interventions=None,
            all_interventions=False,
            format="svg",
            output_dir=directory,
        )
        for key, value in options.items():
            setattr(args, key, value)
        stream = StringIO()
        with redirect_stdout(stream):
            plot.run(args)
        return stream.getvalue()

    def test_selector_uses_visit_and_serialized_dictionary_key(self):
        """Integer dictionary keys become strings on disk; lists stay indexed."""
        selector = {"site": "here", "occurrence": 1, "path": [2, 0]}
        trace = {
            "events": [
                {"site": "here", "occurrence": 0, "value": {"2": [98]}},
                {"site": "here", "occurrence": 1, "value": {"2": [17]}},
            ]
        }
        self.assertEqual(
            plot.read_roles({"mapping": [["scores", [selector]]]}, trace),
            {"scores": [17]},
        )

    def test_exports_separate_ordinary_and_selected_patches_for_both_depths(self):
        """One- and two-step reports support user-selected intervention columns."""
        for fixture, case in (
            ("one_step", "validation-0-0"),
            ("conditional", "validation-0-1"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                self.render(fixture, case, output, interventions=[1, 3])
                self.assertEqual(
                    {p.name for p in output.iterdir()},
                    {"ordinary-tree.svg", "interventions-1-3.svg"},
                )
                self.assertIn(
                    "Program observation",
                    (output / "interventions-1-3.svg").read_text(encoding="utf-8"),
                )

    def test_unresolved_and_unpatched_cases_do_not_invent_evidence(self):
        """A reactive route has no tree; an unprobed supported case has no comparison."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            message = self.render("conditional", "validation-0-0", output)
            self.assertIn("unresolved", message)
            self.assertFalse(list(output.iterdir()))
            message = self.render("conditional", "validation-1-1", output)
            self.assertIn("No paired interventions", message)
            self.assertEqual([p.name for p in output.iterdir()], ["ordinary-tree.svg"])

    def test_comparison_uses_observed_values_even_if_prediction_differs(self):
        """A hypothetical mismatch must remain visible instead of copying prediction."""
        report = json.loads(
            (self.root / "one_step/report.json").read_text(encoding="utf-8")
        )
        row = plot.pick_case(report, None, None)
        explanation = report["frozen_explanations"][0]
        traces = json.loads(
            (self.root / "one_step/ordinary-traces.json").read_text(encoding="utf-8")
        )
        roles = plot.read_roles(explanation, traces[0])
        probes = report["validation"]["explanations"][0]["interventions"]
        num, record = plot.selected_probes(probes, None, False)[0]
        other_action = 1 - record["predicted_output"]
        record = {
            **record,
            "observed_output": other_action,
            "passed": False,
            "observed_roles": {**record["observed_roles"], "scores": [99.75, -99.75]},
        }
        figure = plot.comparison_figure(
            report, row, explanation, roles, [(num, record)]
        )
        predicted, observed = figure.axes
        self.assertIn("failed check", observed.get_title())
        self.assertIn(f"Action {other_action}", observed.texts[0].get_text())
        self.assertNotEqual(predicted.texts[0].get_text(), observed.texts[0].get_text())
        self.assertTrue(any("Score 99.75" in t.get_text() for t in observed.texts))
        self.assertFalse(any("Score 99.75" in t.get_text() for t in predicted.texts))
