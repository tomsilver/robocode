"""Focused tests for the stdlib experiment results viewer."""

# These tests intentionally exercise the viewer's private, stdlib-only helpers.
# pylint: disable=protected-access

from __future__ import annotations

import inspect
import json
import subprocess
import threading
from pathlib import Path

import pytest

from experiments import results_viewer as viewer
from robocode.utils import approach_history
from robocode.utils.episode import run_episode


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)


def _run(path: Path) -> viewer.RunInfo:
    return viewer.RunInfo(
        run_id="demo",
        path=path,
        approach="agentic",
        environment="motion2d_easy",
        primitives="none",
        seed=7,
        budget=5.0,
        num_eval_tasks=3,
        per_instance=False,
    )


def test_sync_drive_and_refresh_syncs_before_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The refresh action pulls Drive before rebuilding the run index."""
    events: list[str] = []

    class _Report:
        downloaded = 1
        unchanged = 2
        removed = 0
        ignored = 3

    class _Sync:
        @staticmethod
        def sync() -> _Report:
            """Record and return one fake Drive synchronization."""
            events.append("sync")
            return _Report()

    def _discover(_root: Path) -> dict[str, viewer.RunInfo]:
        events.append("discover")
        return {}

    monkeypatch.setattr(viewer, "DRIVE_SYNC", _Sync())
    monkeypatch.setattr(viewer, "_discover_runs", _discover)
    monkeypatch.setattr(viewer.SCAN, "root", tmp_path)

    report = viewer.sync_drive_and_refresh()

    assert events == ["sync", "discover"]
    assert report == {"downloaded": 1, "unchanged": 2, "removed": 0, "ignored": 3}


def _assistant(subject: str, tokens: tuple[int, int]) -> dict:
    return {
        "type": "assistant",
        "message": {
            "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
            "content": [
                {"type": "thinking", "thinking": f"Reasoning for {subject}"},
                {
                    "type": "tool_use",
                    "name": "Bash",
                    "input": {
                        "command": f"git add approach.py && git commit -m '{subject}'"
                    },
                },
            ],
        },
    }


def test_preview_step_limit_accepts_custom_positive_integers() -> None:
    """A render request may override the viewer's 100-step preview default."""
    assert viewer._preview_step_limit(None) == viewer.PREVIEW_STEPS
    assert viewer._preview_step_limit(1) == 1
    assert viewer._preview_step_limit("250") == 250


def test_preview_progress_uses_a_bounded_card_layout() -> None:
    """Preview controls and their live status stay within narrow episode cards."""
    assert ".previewctl{display:grid;grid-template-columns:44px minmax(0,1fr)" in (
        viewer.APP_CSS
    )
    assert ".previewctl .render-status{grid-column:1/-1" in viewer.APP_CSS
    assert ".render-status{width:100%;min-width:0;box-sizing:border-box" in (
        viewer.APP_CSS
    )


def test_index_filters_are_restored_from_the_navigation_hash() -> None:
    """Run navigation retains a bookmarkable filtered index destination."""
    assert 'let ALL=[], FILTER={}, INDEX_HASH="#/index"' in viewer.APP_JS
    assert "function indexHash(){" in viewer.APP_JS
    assert "function restoreFilters(hash){" in viewer.APP_JS
    assert "location.hash=indexHash()" in viewer.APP_JS
    assert "restoreFilters(hash);INDEX_HASH=indexHash()" in viewer.APP_JS
    assert '<a id="brand" href="#/index"' in viewer.INDEX_HTML


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "1.5", "many"])
def test_preview_step_limit_rejects_non_positive_or_fractional_values(value) -> None:
    """Invalid custom horizons are rejected before a render is queued."""
    with pytest.raises(ValueError, match="positive integer"):
        viewer._preview_step_limit(value)


def test_snapshots_include_effort_and_replay_progress(tmp_path: Path) -> None:
    """Git snapshots include stream effort and persisted replay outcomes."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    _git(sandbox, "init")
    _git(sandbox, "config", "user.email", "test@example.com")
    _git(sandbox, "config", "user.name", "Test")
    (sandbox / "README.md").write_text("setup\n")
    _git(sandbox, "add", "README.md")
    _git(sandbox, "commit", "-m", "initial setup")

    for subject, source in (
        ("first idea", "x = 1\n"),
        ("fix collision", "x = 2\ny = 3\n"),
    ):
        (sandbox / "approach.py").write_text(source)
        _git(sandbox, "add", "approach.py")
        _git(sandbox, "commit", "-m", subject)

    stream = [
        _assistant("first idea", (100, 20)),
        _assistant("fix collision", (150, 30)),
    ]
    (tmp_path / "stream.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in stream)
    )
    version_dir = tmp_path / "approach_history" / "v001"
    version_dir.mkdir(parents=True)
    (version_dir / "episodes.json").write_text(
        json.dumps(
            {
                "2": {
                    "episode_index": 2,
                    "seed": 123,
                    "solved": False,
                    "crashed": True,
                    "error": "ValueError: collision",
                }
            }
        )
    )

    snapshots = viewer._snapshots(_run(tmp_path))

    assert [s["message"] for s in snapshots] == ["first idea", "fix collision"]
    assert [s["effort"]["tokens"] for s in snapshots] == [120, 180]
    assert snapshots[1]["effort"]["additions"] == 2
    assert snapshots[1]["evaluation"]["solve_rate"] == 0.0
    assert snapshots[1]["evaluation"]["failures"][0]["episode_index"] == 2
    assert snapshots[1]["evaluation"]["failures"][0]["outcome"] == "crashed"


def test_concurrent_snapshot_requests_share_git_work(
    tmp_path: Path, monkeypatch
) -> None:
    """The two history requests made by a run page build snapshots only once."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    _git(sandbox, "init")
    _git(sandbox, "config", "user.email", "test@example.com")
    _git(sandbox, "config", "user.name", "Test")
    (sandbox / "approach.py").write_text("x = 1\n")
    _git(sandbox, "add", "approach.py")
    _git(sandbox, "commit", "-m", "first idea")

    calls = 0
    original = viewer._git

    def counted_git(path: Path, *args: str):
        nonlocal calls
        calls += 1
        return original(path, *args)

    monkeypatch.setattr(viewer, "_git", counted_git)
    run = _run(tmp_path)
    barrier = threading.Barrier(3)
    results = []

    def load() -> None:
        barrier.wait()
        results.append(viewer._snapshots(run))

    threads = [threading.Thread(target=load) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert calls == 3  # log, cat-file, and numstat for one shared build


def test_replay_uses_local_source_and_hydra_add_or_override(tmp_path: Path) -> None:
    """Replays use this checkout and support configs without load/output fields."""
    assert Path(inspect.getfile(run_episode)).is_relative_to(viewer.SRC_ROOT)
    overrides = viewer._replay_overrides(_run(tmp_path), "/tmp/load", "/tmp/out")
    assert "++approach.load_dir=/tmp/load" in overrides
    assert "++approach.output_dir=/tmp/out" in overrides


def test_run_detail_exposes_reproducible_episode_seeds(tmp_path: Path) -> None:
    """Run details reconstruct the exact deterministic evaluation seeds."""
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "solve_rate": 0.5,
                "per_episode": [
                    {"solved": False, "num_steps": 10},
                    {"solved": True, "num_steps": 2},
                ],
            }
        )
    )

    detail = viewer._run_detail(_run(tmp_path))

    assert len(detail["episodes"]) == 2
    # Seeds are strings: they run to 2**63 and a JSON number loses exactness
    # past 2**53 once the browser parses it.
    assert all(isinstance(e["seed"], str) for e in detail["episodes"])
    assert all(int(e["seed"]) >= 0 for e in detail["episodes"])
    assert detail["episodes"][0]["seed"] != detail["episodes"][1]["seed"]

    final = viewer._final_evaluation(_run(tmp_path))
    assert final is not None
    assert final["evaluated"] == 2
    assert final["solve_rate"] == 0.5
    assert final["source"] == "final run"
    assert final["episodes"]["0"]["outcome"] == "failure"


def test_record_history_episode_merges_existing_results(tmp_path: Path) -> None:
    """Recording a replay preserves metrics for other episode indices."""
    run = _run(tmp_path)
    viewer._record_history_episode(run, 0, 1, {"episode_index": 1, "solved": False})
    viewer._record_history_episode(run, 0, 2, {"episode_index": 2, "solved": True})

    records = viewer._history_episode_records(run, 0)

    assert set(records) == {1, 2}
    assert viewer._history_evaluation(run, 0)["solve_rate"] == 0.5


def test_seed_mentions_recovers_agent_test_curriculum() -> None:
    """Literal assignments, explicit lists, and ranges yield training seeds."""
    command = """
for seed in [0, 1, 2, 42]:
    env.reset(seed=seed)
env.reset(seed=17)
for seed in range(20, 50):
    run(seed)
seeds = [(5, 0.1, 0.2), (9, 0.3, 0.4)]
"""

    mentions = viewer._seed_mentions(command)

    assert {0, 1, 2, 5, 9, 17, 20, 42, 49} <= set(mentions)
    assert mentions[17] == {"literal"}
    assert mentions[42] == {"list", "range"}
    assert mentions[49] == {"range"}


def test_object_count_audit_detects_held_out_training_probe(tmp_path: Path) -> None:
    """Explicit training probes outside design_counts are reported as leakage."""
    hydra = tmp_path / ".hydra"
    hydra.mkdir()
    (hydra / "config.yaml").write_text("""environment:
  design_counts:
  - 0
  - 1
  - 2
  eval_counts:
  - 0
  - 1
  - 2
  - 3
  - 4
seed: 7
""")
    source = """
for obj_count in [0, 1, 2, 3]:
    for seed in [0, 1, 42]:
        env.reset(seed=seed, options={"object_count": obj_count})
"""
    (tmp_path / "stream.jsonl").write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Write",
                            "input": {"file_path": "test.py", "content": source},
                        }
                    ]
                },
            }
        )
        + "\n"
    )

    training = viewer._training_seed_info(_run(tmp_path))

    assert training["object_count_audit"] == {
        "status": "violation",
        "design_counts": [0, 1, 2],
        "eval_counts": [0, 1, 2, 3, 4],
        "held_out_counts": [3, 4],
        "observed_counts": [0, 1, 2, 3],
        "violation_counts": [3],
        "other_out_of_domain_counts": [],
        "unresolved_expressions": [],
        "evidence_complete": True,
    }
    seed_42 = next(item for item in training["seeds"] if item["seed"] == 42)
    assert seed_42["object_counts"] == [0, 1, 2, 3]


def test_object_count_mentions_resolves_seed_count_case_table() -> None:
    """Count variables backed by diagnostic case tuples are resolved."""
    source = """
test_cases = [(42, 1), (42, 3), (0, 7)]
for seed, count in test_cases:
    env.reset(seed=seed, options={"object_count": count})
"""

    mentions = viewer._object_count_mentions(source)

    assert mentions == {"counts": [1, 3, 7], "unresolved": []}


def test_generation_time_breakdown_uses_api_wait_and_wall_time() -> None:
    """Claude API wait is split from the remaining local generation time."""
    breakdown = viewer._generation_time_breakdown(
        {
            "gen_wall_time_s": 100.0,
            "gen_cli_duration_ms": 95_000,
            "gen_cli_duration_api_ms": 75_000,
        }
    )

    assert breakdown is not None
    assert breakdown["claude_s"] == 75.0
    assert breakdown["experiments_tools_s"] == 25.0
    assert breakdown["claude_fraction"] == 0.75
    assert breakdown["basis"] == "generation wall time"

    instrumented = viewer._generation_time_breakdown(
        {
            "gen_wall_time_s": 100.0,
            "gen_model_wait_time_s": 60.0,
            "gen_experiment_time_s": 25.0,
            "gen_other_tool_time_s": 10.0,
        }
    )
    assert instrumented is not None
    assert instrumented["instrumented"]
    assert instrumented["experiments_tools_fraction"] == 0.25
    assert instrumented["other_s"] == 15.0  # includes uninstrumented startup


def test_discovery_labels_backend_model(tmp_path: Path) -> None:
    """Runs carry a short model label from approach.backend.model, if any."""

    def make_run(name: str, backend_block: str) -> None:
        hydra = tmp_path / name / ".hydra"
        hydra.mkdir(parents=True)
        (hydra / "config.yaml").write_text(
            f"approach:\n{backend_block}"
            "  _target_: robocode.approaches.agentic_approach.AgenticApproach\n"
            "seed: 42\nnum_eval_tasks: 3\n"
        )
        (hydra / "overrides.yaml").write_text("- seed=42\n")
        (tmp_path / name / "results.json").write_text("{}")

    make_run("pinned", "  backend:\n    backend: claude\n    model: claude-opus-5\n")
    make_run("alias", "  backend:\n    backend: claude\n    model: sonnet\n")
    make_run("no_llm", "  backend: null\n")

    runs = viewer._discover_runs(tmp_path)
    assert runs["pinned"].model == "opus-5"
    assert runs["alias"].model == "sonnet-4.6"
    assert runs["no_llm"].model is None


def test_history_evaluation_cancel_persists_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cancel unwinds the job instead of recording fabricated crash records."""
    run = _run(tmp_path)
    snapshot = {"commit_hash": "abc", "version": 0, "evaluation": {"evaluated": 0}}
    final = {"commit_hash": "def", "version": 1, "evaluation": {"evaluated": 2}}
    monkeypatch.setattr(viewer, "_snapshots", lambda _run: [snapshot, final])
    monkeypatch.setattr(viewer, "_eval_seeds", lambda _run: [1, 2])
    monkeypatch.setattr(
        viewer, "_results", lambda _run: {"per_episode": [{"solved": True}] * 2}
    )

    def _cancelled_export(*_args: object) -> None:
        raise viewer._Cancelled

    monkeypatch.setattr(approach_history, "_export_snapshot", _cancelled_export)

    with pytest.raises(viewer._Cancelled):
        viewer._evaluate_history(run, viewer.Job(job_id="j"), epoch=0)

    assert not (tmp_path / "approach_history" / "v000" / "episodes.json").exists()
