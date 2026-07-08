"""Tests for the plot_scaling experiment script.

experiments/ is a scripts directory, not an installed package, so the module is loaded
from its path. Its helpers are exercised directly: source tracking and label
disambiguation in collect/series, per-count aggregation in by_count, and that
plot_environment writes a PNG.
"""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "experiments" / "plot_scaling.py"
_SPEC = importlib.util.spec_from_file_location("plot_scaling", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
plot_scaling: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plot_scaling)


def _write_run(
    root: Path,
    name: str,
    approach: str,
    episodes: list[dict[str, Any]],
    environment: str = "stickbutton2d_generalized",
) -> Path:
    """Write a minimal run dir (results.json + .hydra sidecar) and return it."""
    run_dir = root / name
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text("seed: 0\n", encoding="utf-8")
    (hydra_dir / "overrides.yaml").write_text(
        f"- environment={environment}\n- approach={approach}\n", encoding="utf-8"
    )
    (run_dir / "results.json").write_text(
        json.dumps({"per_episode": episodes}), encoding="utf-8"
    )
    return run_dir


def _solved_eps(pattern: dict[int, bool]) -> list[dict[str, Any]]:
    """One episode per (object_count -> solved) entry."""
    return [{"object_count": c, "solved": s} for c, s in pattern.items()]


def test_collect_tracks_source_per_run(tmp_path: Path) -> None:
    """Two runs of the same approach stay under distinct source keys."""
    a = _write_run(tmp_path, "prog", "agentic", _solved_eps({1: True}))
    b = _write_run(tmp_path, "noprims", "agentic", _solved_eps({1: False}))
    c = _write_run(tmp_path, "planner", "bilevel_planning", _solved_eps({1: True}))

    data = plot_scaling.collect([a, b, c])

    assert set(data) == {"stickbutton2d_generalized"}
    by_approach = data["stickbutton2d_generalized"]
    assert set(by_approach) == {"agentic", "bilevel_planning"}
    assert set(by_approach["agentic"]) == {"prog", "noprims"}
    assert set(by_approach["bilevel_planning"]) == {"planner"}


def test_series_splits_colliding_approaches() -> None:
    """One source keeps the plain name; several are suffixed with the source."""
    by_approach = {
        "agentic": {"prog": _solved_eps({1: True}), "noprims": _solved_eps({1: False})},
        "bilevel_planning": {"planner": _solved_eps({1: True})},
    }

    series = plot_scaling.series(by_approach)

    assert set(series) == {
        "agentic (prog)",
        "agentic (noprims)",
        "bilevel_planning",
    }
    assert series["agentic (prog)"] == by_approach["agentic"]["prog"]


def test_series_single_source_keeps_plain_name() -> None:
    """A lone run of an approach is labelled by the bare approach name."""
    episodes = _solved_eps({1: True, 2: False})
    series = plot_scaling.series({"agentic": {"prog": episodes}})
    assert series == {"agentic": episodes}


def test_seeds_under_one_dir_pool(tmp_path: Path) -> None:
    """Sub-runs found under a single passed dir pool into one source, one line."""
    root = tmp_path / "sweep"
    _write_run(root, "seed0", "agentic", _solved_eps({1: True}))
    _write_run(root, "seed1", "agentic", _solved_eps({1: False}))

    by_approach = plot_scaling.collect([root])["stickbutton2d_generalized"]
    series = plot_scaling.series(by_approach)

    assert set(by_approach["agentic"]) == {"sweep"}
    assert set(series) == {"agentic"}
    assert len(series["agentic"]) == 2


def test_by_count_rate_and_mean() -> None:
    """Rate reduces booleans to a solved fraction; mean averages a numeric field."""
    episodes = [
        {"object_count": 1, "solved": True, "planning_time": 2.0},
        {"object_count": 1, "solved": False, "planning_time": 4.0},
        {"object_count": 2, "solved": True, "planning_time": 9.0},
    ]
    counts, rate = plot_scaling.by_count(episodes, "solved", reducer="rate")
    assert counts == [1, 2]
    assert rate == pytest.approx([0.5, 1.0])

    counts, mean = plot_scaling.by_count(episodes, "planning_time")
    assert counts == [1, 2]
    assert mean == pytest.approx([3.0, 9.0])


def test_plot_environment_writes_png(tmp_path: Path) -> None:
    """A full render writes a non-empty PNG for the environment."""
    series = {
        "agentic": _solved_eps({1: True, 2: True, 3: False}),
        "bilevel_planning": [
            {
                "object_count": 1,
                "solved": True,
                "planning_time": 3.0,
                "plan_found": True,
            },
            {
                "object_count": 3,
                "solved": False,
                "planning_time": 30.0,
                "plan_found": False,
            },
        ],
    }
    out = tmp_path / "scaling.png"

    plot_scaling.plot_environment("stickbutton2d_generalized", series, out)

    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
