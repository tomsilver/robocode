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

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from robocode.utils import rate_limit
from robocode.utils.backends import DEFAULT_BACKEND_CFG
from robocode.utils.backends.claude import ClaudeBackend
from robocode.utils.rate_limit import run_with_rate_limit_retry
from robocode.utils.sandbox import SandboxConfig, SandboxResult

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


_PARALLEL_FAKE_CLAUDE = """\
#!/usr/bin/env python3
import json, os, pathlib, sys, time

cwd = pathlib.Path.cwd()
counter = cwd / "fake_counter.txt"
n = (int(counter.read_text()) if counter.exists() else 0) + 1
counter.write_text(str(n))
prompt = sys.argv[sys.argv.index("-p") + 1]
marker = pathlib.Path(os.environ["CLAUDE_CONFIG_DIR"]) / "resume_marker.txt"


def emit(obj):
    print(json.dumps(obj))


if n == 1:
    marker.write_text(prompt)
    gate = pathlib.Path(os.environ["FAKE_CLAUDE_PARALLEL_GATE"])
    (gate / f"ready-{prompt}").write_text("ready")
    deadline = time.monotonic() + 10
    while len(list(gate.glob("ready-*"))) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    if len(list(gate.glob("ready-*"))) < 2:
        emit({"type": "result", "subtype": "error", "is_error": True,
              "result": "parallel test timed out", "total_cost_usd": 0.0,
              "modelUsage": {}})
        sys.exit(1)
    emit({"type": "result", "subtype": "error", "is_error": True,
          "result": "You've hit your limit resets 3am", "total_cost_usd": 0.5,
          "modelUsage": {}})
    sys.exit(1)

resumed_own_session = (
    "--continue" in sys.argv and marker.exists() and marker.read_text() == prompt
)
if not resumed_own_session:
    emit({"type": "result", "subtype": "error", "is_error": True,
          "result": "resumed another run", "total_cost_usd": 0.0,
          "modelUsage": {}})
    sys.exit(1)

pathlib.Path("approach.py").write_text(
    f"# {prompt}\\nclass GeneratedApproach:\\n    pass\\n"
)
emit({"type": "result", "subtype": "success", "is_error": False,
      "result": "done", "total_cost_usd": 1.0, "modelUsage": {}})
"""


_OUTPUT_LIMIT_FAKE_CLAUDE = """\
#!/usr/bin/env python3
import json, os, pathlib, sys

counter = pathlib.Path.cwd() / "fake_counter.txt"
n = (int(counter.read_text()) if counter.exists() else 0) + 1
counter.write_text(str(n))
marker = pathlib.Path(os.environ["CLAUDE_CONFIG_DIR"]) / "output_limit_marker.txt"
prompt = sys.argv[sys.argv.index("-p") + 1]

if n == 1:
    marker.write_text("interrupted")
    print(json.dumps({"type": "assistant", "message": {"content": [{
        "type": "text", "text": "API Error: Claude's response exceeded the 32768 "
        "output token maximum. To configure this behavior, set the "
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable."
    }]}}))
    print(json.dumps({"type": "result", "subtype": "error_during_execution",
                      "is_error": True, "result": "response exceeded the 32768 "
                      "output token maximum", "total_cost_usd": 0.5,
                      "modelUsage": {}}))
    sys.exit(1)

resumed = (
    "--continue" in sys.argv and marker.exists() and "Be concise" in prompt
)
if not resumed:
    print(json.dumps({"type": "result", "subtype": "error", "is_error": True,
                      "result": "not resumed concisely", "total_cost_usd": 0.0,
                      "modelUsage": {}}))
    sys.exit(1)

pathlib.Path("approach.py").write_text("class GeneratedApproach:\\n    pass\\n")
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "result": "done", "total_cost_usd": 0.25,
                  "modelUsage": {}}))
"""


_PROMPT_TOO_LONG_FAKE_CLAUDE = """\
#!/usr/bin/env python3
import json, os, pathlib, sys

counter = pathlib.Path.cwd() / "fake_counter.txt"
n = (int(counter.read_text()) if counter.exists() else 0) + 1
counter.write_text(str(n))
marker = pathlib.Path(os.environ["CLAUDE_CONFIG_DIR"]) / "context_marker.txt"
prompt = sys.argv[sys.argv.index("-p") + 1]

if n == 1:
    marker.write_text("oversized")
    print(json.dumps({"type": "assistant", "message": {"content": [{
        "type": "text", "text": "Prompt is too long"
    }]}}))
    print(json.dumps({"type": "result", "subtype": "error_during_execution",
                      "is_error": True, "result": "Prompt is too long",
                      "total_cost_usd": 0.5, "modelUsage": {}}))
    sys.exit(1)

resumed = "--continue" in sys.argv and marker.exists()
if n == 2 and resumed and prompt.startswith("/compact "):
    print(json.dumps({"type": "system", "subtype": "compact_boundary",
                      "compact_metadata": {"trigger": "manual",
                                           "pre_tokens": 1000}}))
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                      "result": "Conversation compacted", "total_cost_usd": 0.25,
                      "modelUsage": {}}))
    sys.exit(0)

if n != 3 or not resumed or "compacted conversation" not in prompt:
    print(json.dumps({"type": "result", "subtype": "error", "is_error": True,
                      "result": "invalid compaction recovery", "total_cost_usd": 0.0,
                      "modelUsage": {}}))
    sys.exit(1)

pathlib.Path("approach.py").write_text("class GeneratedApproach:\\n    pass\\n")
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "result": "done", "total_cost_usd": 0.25,
                  "modelUsage": {}}))
"""


def test_end_to_end_resume_with_fake_cli(tmp_path: Path, monkeypatch) -> None:
    """A rate-limited first attempt is resumed and completes on retry."""
    fake = tmp_path / "fake_claude.py"
    fake.write_text(_FAKE_CLAUDE)
    fake.chmod(0o755)
    counter = tmp_path / "counter.txt"
    monkeypatch.setenv("ROBOCODE_CLAUDE_CMD", str(fake))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-stable-token")
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


def test_parallel_runs_resume_only_their_own_session(
    tmp_path: Path, monkeypatch
) -> None:
    """Concurrent rate-limited runs have disjoint --continue lookup stores."""
    fake = tmp_path / "parallel_fake_claude.py"
    fake.write_text(_PARALLEL_FAKE_CLAUDE)
    fake.chmod(0o755)
    gate = tmp_path / "gate"
    gate.mkdir()
    monkeypatch.setenv("ROBOCODE_CLAUDE_CMD", str(fake))
    monkeypatch.setenv("FAKE_CLAUDE_PARALLEL_GATE", str(gate))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-stable-token")
    monkeypatch.setattr(rate_limit.time, "sleep", lambda _s: None)

    configs = [
        SandboxConfig(
            sandbox_dir=tmp_path / f"sandbox-{name}",
            prompt=name,
            output_filename="approach.py",
            max_budget_usd=5.0,
            mcp_tools=(),
        )
        for name in ("alpha", "beta")
    ]

    def run(config: SandboxConfig) -> SandboxResult:
        """Run one independently configured generation."""
        return run_with_rate_limit_retry(
            None, config, backend=ClaudeBackend(DEFAULT_BACKEND_CFG)
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, configs))

    assert all(result.success for result in results)
    for config, result in zip(configs, results, strict=True):
        assert result.output_file is not None
        assert result.output_file.read_text().startswith(f"# {config.prompt}\n")
        assert (config.sandbox_dir / "fake_counter.txt").read_text() == "2"
        assert result.generation_metrics is not None
        assert result.generation_metrics.rate_limit_retries == 1


def test_end_to_end_resume_after_output_token_limit(
    tmp_path: Path, monkeypatch
) -> None:
    """The observed Claude output-limit error resumes with remaining budget."""
    fake = tmp_path / "output_limit_fake_claude.py"
    fake.write_text(_OUTPUT_LIMIT_FAKE_CLAUDE)
    fake.chmod(0o755)
    monkeypatch.setenv("ROBOCODE_CLAUDE_CMD", str(fake))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-stable-token")
    config = SandboxConfig(
        sandbox_dir=tmp_path / "sandbox-output-limit",
        prompt="solve it",
        output_filename="approach.py",
        max_budget_usd=2.0,
        max_turns=10,
        mcp_tools=(),
    )

    final = run_with_rate_limit_retry(
        None, config, backend=ClaudeBackend(DEFAULT_BACKEND_CFG)
    )

    assert final.success
    assert (config.sandbox_dir / "fake_counter.txt").read_text() == "2"
    assert final.generation_metrics is not None
    assert final.generation_metrics.output_token_retries == 1
    assert final.generation_metrics.aborted_cost_usd == 0.5


def test_end_to_end_compact_then_resume_after_prompt_too_long(
    tmp_path: Path, monkeypatch
) -> None:
    """The exact context error triggers /compact before continuing work."""
    fake = tmp_path / "prompt_too_long_fake_claude.py"
    fake.write_text(_PROMPT_TOO_LONG_FAKE_CLAUDE)
    fake.chmod(0o755)
    monkeypatch.setenv("ROBOCODE_CLAUDE_CMD", str(fake))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-stable-token")
    config = SandboxConfig(
        sandbox_dir=tmp_path / "sandbox-prompt-too-long",
        prompt="solve it",
        output_filename="approach.py",
        max_budget_usd=2.0,
        max_turns=10,
        mcp_tools=(),
    )

    final = run_with_rate_limit_retry(
        None, config, backend=ClaudeBackend(DEFAULT_BACKEND_CFG)
    )

    assert final.success
    assert (config.sandbox_dir / "fake_counter.txt").read_text() == "3"
    assert final.generation_metrics is not None
    assert final.generation_metrics.prompt_too_long_retries == 1
    assert final.generation_metrics.aborted_cost_usd == 0.75
