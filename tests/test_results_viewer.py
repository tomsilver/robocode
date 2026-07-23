"""Focused tests for the stdlib experiment results viewer."""

# These tests intentionally exercise the viewer's private, stdlib-only helpers.
# pylint: disable=protected-access

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from experiments import results_viewer as viewer


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)


def _run(path: Path) -> viewer.RunInfo:
    return viewer.RunInfo(
        run_id="demo",
        path=path,
        approach="agentic",
        environment="motion2d_easy",
        seed=7,
        budget=5.0,
        num_eval_tasks=3,
        per_instance=False,
    )


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
    assert all(isinstance(e["seed"], int) for e in detail["episodes"])
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
