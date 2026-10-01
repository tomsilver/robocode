"""Tests for the per-checkpoint re-evaluation pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from experiments.reeval import collect, evaluate, prepare, worker
from experiments.reeval.store import (
    Task,
    checkpoint_trees,
    final_tree,
    list_tasks,
    load_run,
)

_ENV = {"_target_": "some.Env", "eval_counts": [1, 2]}


def _commit(sandbox: Path, files: dict[str, str], message: str) -> None:
    for name, text in files.items():
        (sandbox / name).write_text(text)
    subprocess.run(["git", "add", "-A"], cwd=sandbox, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message],
        cwd=sandbox,
        check=True,
    )


def _make_run(root: Path, key: str = "exp__a") -> Path:
    """Replicate dir with four commits: setup, v1, v2, and a revert to v1."""
    replicate = root / key / "2026-01-01_00-00-00" / "replicate_42"
    sandbox = replicate / "sandbox"
    sandbox.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=sandbox, check=True)
    _commit(sandbox, {"CLAUDE.md": "hi"}, "initial sandbox setup")
    _commit(sandbox, {"approach.py": "v1"}, "v1")
    _commit(sandbox, {"approach.py": "v2"}, "v2")
    _commit(sandbox, {"approach.py": "v1"}, "back to v1")
    (replicate / ".hydra").mkdir()
    config = {
        "approach": {"blackbox": True, "blackbox_strict": True},
        "environment": _ENV,
        "replicate_seed": 42,
        "max_steps": 1000,
        "num_eval_tasks": 2,
        "eval_timeout": 60,
        "eval_seed": 7,
    }
    (replicate / ".hydra" / "config.yaml").write_text(yaml.safe_dump(config))
    results = {"solve_rate": 0.5, "per_episode": [{"solved": True}, {"solved": False}]}
    (replicate / "results.json").write_text(json.dumps(results))
    (replicate / "run_experiment.log").write_text(
        "[2026-01-01 00:10:00,000][__main__][INFO] - Training complete; starting 2 "
        "held-out evaluation episodes\n"
        "[2026-01-01 00:12:30,000][__main__][INFO] - Mean reward: 1\n"
    )
    return replicate


@pytest.fixture(name="store")
def _store(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    assert prepare.add_run(_make_run(tmp_path / "results"), store) == "added"
    return store


def test_prepare_records_commits_and_settings(tmp_path: Path, store: Path) -> None:
    """The store keeps the history, eval settings and original outcomes, not the
    seed."""
    run = load_run(store, "exp__a__2026-01-01_00-00-00__replicate_42")
    assert [c["subject"] for c in run["commits"]][0] == "initial sandbox setup"
    assert [c["has_approach"] for c in run["commits"]] == [False, True, True, True]
    assert run["environment"] == _ENV
    assert run["blackbox_strict"] is True
    assert run["original"] == {"solve_rate": 0.5, "solved": [True, False]}
    assert run["est_eval_seconds"] == 150.0
    assert "eval_seed" not in json.dumps(run)
    replicate = tmp_path / "results" / "exp__a" / "2026-01-01_00-00-00" / "replicate_42"
    assert prepare.add_run(replicate, store) == "present"


def test_identical_trees_are_evaluated_once(store: Path) -> None:
    """A commit that restores an earlier tree adds no checkpoint."""
    run = load_run(store, "exp__a__2026-01-01_00-00-00__replicate_42")
    trees = checkpoint_trees(run)
    assert len(trees) == 2
    assert final_tree(run) == trees[0]
    assert len(list_tasks(store, "all")) == 2
    assert [t.tree for t in list_tasks(store, "final")] == [trees[0]]


def test_prepare_skips_runs_without_history(tmp_path: Path) -> None:
    """Planner and GenPlan runs have no sandbox repository."""
    replicate = tmp_path / "exp" / "ts" / "replicate_1"
    replicate.mkdir(parents=True)
    status = prepare.add_run(replicate, tmp_path / "store")
    assert status == "skipped: no sandbox git history"


def test_export_tree_from_bundle(tmp_path: Path, store: Path) -> None:
    """A checkpoint is rebuilt from the single-file bundle."""
    key = "exp__a__2026-01-01_00-00-00__replicate_42"
    run = load_run(store, key)
    dest = tmp_path / "x" / "sandbox"
    dest.parent.mkdir()
    evaluate.export_tree(
        store / "runs" / key / "repo.bundle", run["commits"][2]["tree"], dest
    )
    assert (dest / "approach.py").read_text() == "v2"


def test_shards_cover_every_task_once() -> None:
    """LPT sharding is a partition with balanced loads."""
    tasks = [Task(f"r{i}", f"t{i}", float(i % 7 + 1)) for i in range(50)]
    shards = worker.assign_shards(tasks, 4)
    flat = [t for s in shards for t in s]
    assert sorted(flat, key=lambda t: t.tree) == sorted(tasks, key=lambda t: t.tree)
    loads = [sum(t.est_seconds for t in s) for s in shards]
    assert max(loads) - min(loads) <= 7
    assert worker.assign_shards(tasks, 4) == shards


def _fake_harness(results: dict[str, Any] | None) -> list[str]:
    """A stand-in for run_experiment.py that writes ``results`` into hydra.run.dir."""
    script = (
        "import json, pathlib, sys\n"
        "arg = [a for a in sys.argv if a.startswith('hydra.run.dir=')][0]\n"
        "run_dir = arg.split('=', 1)[1]\n"
        "p = pathlib.Path(run_dir); p.mkdir(parents=True)\n"
        f"r = {results!r}\n"
        "if r is None: sys.exit('ImportError: approach.py broke')\n"
        "(p / 'results.json').write_text(json.dumps(r))\n"
    )
    return [sys.executable, "-c", script]


@pytest.mark.parametrize("ok", [True, False])
def test_evaluate_records_outcome(
    tmp_path: Path, store: Path, monkeypatch: pytest.MonkeyPatch, ok: bool
) -> None:
    """Successful runs keep the results without the seed; failures keep the log."""
    results = {"solve_rate": 1.0, "eval_seed": 7, "per_episode": []} if ok else None

    def fake_command(*args: Any) -> list[str]:
        run_dir = args[3]
        return _fake_harness(results) + [f"hydra.run.dir={run_dir}"]

    monkeypatch.setattr(evaluate, "build_command", fake_command)
    key = "exp__a__2026-01-01_00-00-00__replicate_42"
    tree = final_tree(load_run(store, key))
    assert tree is not None
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    record = evaluate.evaluate(Task(key, tree, 1.0), store, scratch, seed=7)
    assert len(record["commits"]) == 2
    assert not list(scratch.iterdir())
    if ok:
        assert record["status"] == "ok"
        assert record["results"] == {"solve_rate": 1.0, "per_episode": []}
    else:
        assert record["status"] == "failed"
        assert "approach.py broke" in record["log_tail"]


def test_command_carries_run_settings(tmp_path: Path, store: Path) -> None:
    """The harness call reuses the run's flags and only evaluates the checkpoint."""
    run = load_run(store, "exp__a__2026-01-01_00-00-00__replicate_42")
    cmd = evaluate.build_command(run, tmp_path / "c", tmp_path / "l", tmp_path / "r", 7)
    assert "approach.blackbox_strict=True" in cmd
    assert f"approach.load_dir={tmp_path / 'l'}" in cmd
    assert "num_eval_tasks=2" in cmd and "eval_seed=7" in cmd


def test_validation_compares_episodes(tmp_path: Path, store: Path) -> None:
    """The final checkpoint is compared episode by episode with the original."""
    key = "exp__a__2026-01-01_00-00-00__replicate_42"
    run = load_run(store, key)
    tree = final_tree(run)
    out = tmp_path / "out"
    (out / key).mkdir(parents=True)
    record = {
        "status": "ok",
        "results": {"solve_rate": 1.0, "per_episode": [{"solved": True}] * 2},
    }
    (out / key / f"{tree}.json").write_text(json.dumps(record))
    row = collect.validation_row(run, out)
    assert row["episodes_compared"] == 2 and row["episodes_agreeing"] == 1
    rows = collect.checkpoint_rows(run, out)
    assert [r["status"] for r in rows] == ["no_approach", "ok", "pending", "ok"]
