"""End-to-end test of the rate-limit resume flow with a fake Claude CLI.

A real usage-cap interruption cannot be triggered on demand, so this swaps the
``claude`` binary (via ``ROBOCODE_CLAUDE_CMD``) for a stand-in that reports a
rate limit on its first invocation and completes on the second. The whole real
path runs unchanged: ``run_with_rate_limit_retry`` -> ``run_agent_in_sandbox``
(subprocess launch, ``CLAUDE_CONFIG_DIR`` redirect, session dir) ->
``parse_stream`` -> rate-limit detection -> resume with ``--continue``.

The stand-in only completes when it is both resumed (``--continue`` is present)
and can see the session state it persisted on the first attempt, so a passing
run proves the flag reaches the CLI and the session store survives the retry.
"""

from pathlib import Path

from robocode.utils import rate_limit
from robocode.utils.backends import DEFAULT_BACKEND_CFG
from robocode.utils.backends.claude import ClaudeBackend
from robocode.utils.rate_limit import run_with_rate_limit_retry
from robocode.utils.sandbox import SandboxConfig

# A fake ``claude``: first call persists a session marker under CLAUDE_CONFIG_DIR
# and reports a usage-cap error; the retry succeeds only when resumed and able to
# read that marker back. A counter outside the sandbox bounds it to two calls.
_FAKE_CLAUDE = """\
#!/usr/bin/env python3
import json, os, pathlib, sys

counter = pathlib.Path(os.environ["FAKE_CLAUDE_COUNTER"])
n = (int(counter.read_text()) if counter.exists() else 0) + 1
counter.write_text(str(n))

marker = pathlib.Path(os.environ["CLAUDE_CONFIG_DIR"]) / "resume_marker.txt"


def emit(obj):
    print(json.dumps(obj))


if n == 1:
    marker.write_text("session-1")
    emit({"type": "result", "subtype": "error", "is_error": True,
          "result": "You've hit your limit resets 3am", "total_cost_usd": 0.5,
          "modelUsage": {"claude": {"inputTokens": 100, "outputTokens": 50,
                                    "cacheReadInputTokens": 0,
                                    "cacheCreationInputTokens": 0}}})
    sys.exit(1)

resumed = "--continue" in sys.argv and marker.exists()
if not resumed:
    emit({"type": "result", "subtype": "error", "is_error": True,
          "result": "not resumed", "total_cost_usd": 0.0, "modelUsage": {}})
    sys.exit(1)

pathlib.Path("approach.py").write_text("class GeneratedApproach:\\n    pass\\n")
emit({"type": "result", "subtype": "success", "is_error": False,
      "result": "done", "total_cost_usd": 1.0,
      "modelUsage": {"claude": {"inputTokens": 300, "outputTokens": 100,
                                "cacheReadInputTokens": 0,
                                "cacheCreationInputTokens": 0}}})
"""


def test_end_to_end_resume_with_fake_cli(tmp_path: Path, monkeypatch) -> None:
    """A rate-limited first attempt is resumed and completes on retry."""
    fake = tmp_path / "fake_claude.py"
    fake.write_text(_FAKE_CLAUDE)
    fake.chmod(0o755)
    counter = tmp_path / "counter.txt"
    monkeypatch.setenv("ROBOCODE_CLAUDE_CMD", str(fake))
    monkeypatch.setenv("FAKE_CLAUDE_COUNTER", str(counter))
    monkeypatch.setattr(rate_limit.time, "sleep", lambda _s: None)

    config = SandboxConfig(
        sandbox_dir=tmp_path / "sandbox",
        prompt="solve it",
        output_filename="approach.py",
        model="sonnet",
        max_budget_usd=5.0,
        mcp_tools=(),
    )
    final = run_with_rate_limit_retry(
        None, config, backend=ClaudeBackend(DEFAULT_BACKEND_CFG)
    )

    assert final.success
    assert final.output_file == config.sandbox_dir / "approach.py"
    assert final.generation_metrics is not None
    assert final.generation_metrics.rate_limit_retries == 1
    assert final.generation_metrics.aborted_cost_usd == 0.5
    assert counter.read_text() == "2"  # exactly one rate-limit, then one resume
