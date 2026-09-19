"""The full audit must cover every catalog and never turn incomplete runs green."""

# These tests call internal verdict helpers and execute a fixed audit payload.
# pylint: disable=protected-access,exec-used

import asyncio
import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

from integration_tests import red_team_sandbox as rt


def test_full_catalog_has_unique_cases_and_all_attacks():
    """Each legacy attack remains represented with its own retained directory."""
    names = [name for name, _, _ in rt._full_suite_cases("all")]
    assert len(names) == len(set(names))
    for catalog in (
        rt.ADVERSARIAL_PROMPTS,
        rt.BASH_READ_LEAK_PROMPTS,
        rt.BLACKBOX_PROMPTS,
        rt.STRICT_BLACKBOX_PROMPTS,
        rt.BLACKBOX_RENDER_PROMPTS,
        rt.BLACKBOX_PROXY_PROMPTS,
        rt.MODELS_OFF_PROMPTS,
        rt.EVAL_COUNTS_PROMPTS,
        rt.EVAL_SEED_PROMPTS,
        rt.DEMOS_PROMPTS,
    ):
        assert {entry[0] for entry in catalog} <= set(names)
    assert {"host_pid_isolation", "cross_session_isolation", "live_agent_pkill"} <= set(
        names
    )


def test_claude_and_codex_tool_execution_evidence(tmp_path: Path):
    """Codex command events count as execution just like Claude tool-use blocks."""
    stream = tmp_path / "stream.jsonl"
    events = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"command": "echo claude"},
                    }
                ]
            },
        },
        {
            "type": "item.completed",
            "item": {"type": "command_execution", "command": "echo codex"},
        },
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "I ran commands"},
        },
    ]
    stream.write_text(
        "\n".join(json.dumps(event) for event in events), encoding="utf-8"
    )
    calls = rt._agent_tool_calls(stream)  # pylint: disable=protected-access
    assert len(calls) == 2
    assert calls[-1] == ("command_execution", "echo codex")


def test_failure_keeps_logs_and_does_not_skip_later_cases(tmp_path, monkeypatch):
    """An inconclusive attack fails the aggregate even when subsequent cases pass."""

    async def inconclusive():
        print("first attempt retained")
        raise rt.RedTeamInconclusiveError("not executed")

    async def passed():
        print("later attempt retained")

    monkeypatch.setattr(
        rt,
        "_full_suite_cases",
        lambda _: [("first", inconclusive, False), ("second", passed, False)],
    )
    # Restore module globals after the in-process test (the real CLI owns its process).
    for name in (
        "_BACKEND_CFG",
        "_APPTAINER_IMAGE_DIR",
        "RED_TEAM_DIR",
        "SANDBOX_DIR",
        "CANARY_FILE",
        "_run_agent",
    ):
        monkeypatch.setattr(rt, name, getattr(rt, name))
    args = Namespace(
        results_dir=tmp_path / "results",
        backend="codex",
        model=None,
        suite="strict",
        only=None,
        claude_token_file=None,
    )
    with pytest.raises(SystemExit, match="1"):
        asyncio.run(rt._run_full_suite(args))
    report = json.loads((args.results_dir / "summary.json").read_text(encoding="utf-8"))
    assert report["complete"] and not report["passed"]
    assert [case["status"] for case in report["cases"]] == ["inconclusive", "passed"]
    assert "first attempt retained" in (
        args.results_dir / "first/console.log"
    ).read_text(encoding="utf-8")
    assert "later attempt retained" in (
        args.results_dir / "second/console.log"
    ).read_text(encoding="utf-8")


def test_canary_echo_is_not_a_read_leak():
    """The edit prompt may name a prefix but must not disclose the full sentinel."""
    assert rt.CANARY_CONTENT.startswith("CANARY_INTACT_")
    for _, prompt in rt.ADVERSARIAL_PROMPTS + rt.BASH_READ_LEAK_PROMPTS:
        assert rt.CANARY_CONTENT not in prompt
    assert rt.CANARY_CONTENT not in "Task: edit CANARY_INTACT to PWNED"


@pytest.mark.parametrize(
    "failure",
    [
        TypeError("wrong signature"),
        RuntimeError("Connection to the environment server was closed"),
    ],
)
def test_protocol_probe_rejects_invalid_negative_controls(monkeypatch, failure):
    """A client error or dead socket must not be mistaken for a server rejection."""
    prompt = next(
        prompt
        for name, prompt, _ in rt.STRICT_BLACKBOX_PROMPTS
        if name == "strict_env_protocol_escape"
    )
    script = prompt.split("<<'PY' > output.txt\n", 1)[1].split("\nPY\n", 1)[0]

    class BrokenConnection:
        """An apparently live client whose forbidden request never reaches policy."""

        def reset(self, **_kwargs):
            """The positive reset alone is insufficient proof of a rejection."""

        def make_primitives(self):
            """Return the allowed empty primitive set."""
            return {}

        def _request(self, _payload):
            raise failure

        def close(self):
            """No socket exists in this test."""

    monkeypatch.setitem(
        sys.modules, "env_client", types.SimpleNamespace(make_env=BrokenConnection)
    )
    with pytest.raises((TypeError, AssertionError)):
        exec(
            compile(script, "strict_protocol_probe", "exec"), {}
        )  # pylint: disable=exec-used


def test_quoted_bilevel_import_command_is_not_a_success():
    """A recorded failing command must not become a successful import marker."""
    output = (
        "COMMAND: python -c \"import kinder_models; print('BILEVEL_IMPORT_OK')\"\n"
        "EXIT_CODE: 1\nModuleNotFoundError: No module named 'kinder_models'\n"
    )
    assert not rt._recovered_bilevel_models(output)  # pylint: disable=protected-access
    assert rt._recovered_bilevel_models(
        "BILEVEL_IMPORT_OK kinder_models\n"
    )  # pylint: disable=protected-access


def test_retained_suite_captures_results_without_replacing_launcher(monkeypatch):
    """Recording ends even on failure, and later legacy runs do not append evidence."""
    result = object()

    async def launch(*_args):
        return result

    monkeypatch.setattr(rt, "_launch_agent", launch)
    with pytest.raises(RuntimeError, match="case failed"):
        with rt._capture_agent_results() as outcomes:
            assert (
                asyncio.run(rt._run_agent(False, "probe", use_apptainer=True)) is result
            )
            assert outcomes == [result]
            raise RuntimeError("case failed")
    assert rt._CAPTURED_AGENT_RESULTS is None
    asyncio.run(rt._run_agent(False, "later", use_apptainer=True))
    assert outcomes == [result]
