"""Tests for agent backends (Claude, OpenCode)."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig

from robocode.utils.backends import (
    DEFAULT_BACKEND_CFG,
    DEFAULT_OPENCODE_CFG,
    PROVIDERS,
    create_backend,
    firewall_domains_for_provider,
    provider_from_model,
)
from robocode.utils.backends.claude import _RATE_LIMIT_RE, ClaudeBackend
from robocode.utils.backends.opencode import OpenCodeBackend
from robocode.utils.sandbox import SandboxConfig

# ---------------------------------------------------------------------------
# create_backend factory
# ---------------------------------------------------------------------------


class TestCreateBackend:
    """Tests for the create_backend factory function."""

    def test_create_claude_backend(self) -> None:
        """create_backend returns a ClaudeBackend for 'claude'."""
        cfg = DEFAULT_BACKEND_CFG
        backend = create_backend(cfg)
        assert isinstance(backend, ClaudeBackend)

    def test_create_opencode_backend(self) -> None:
        """create_backend returns an OpenCodeBackend for 'opencode'."""
        cfg = DictConfig({"backend": "opencode", "model": "openai/gpt-4o"})
        backend = create_backend(cfg)
        assert isinstance(backend, OpenCodeBackend)

    def test_unknown_backend_raises(self) -> None:
        """Unknown backend name raises ValueError."""
        cfg = DictConfig({"backend": "unknown_llm", "model": "x"})
        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend(cfg)

    def test_backends_satisfy_protocol(self) -> None:
        """Both backends have the required methods."""
        for cfg in (
            DEFAULT_BACKEND_CFG,
            DictConfig({"backend": "opencode", "model": "openai/gpt-4o"}),
        ):
            backend = create_backend(cfg)
            assert hasattr(backend, "build_cli_cmd")
            assert hasattr(backend, "build_env")
            assert hasattr(backend, "setup_sandbox_files")
            assert hasattr(backend, "parse_stream")


# ---------------------------------------------------------------------------
# ClaudeBackend
# ---------------------------------------------------------------------------


class TestClaudeBackend:
    """Tests for the Claude Code CLI backend."""

    def test_build_cli_cmd_basics(self, tmp_path: Path) -> None:
        """CLI command includes the expected flags."""
        config = SandboxConfig(
            sandbox_dir=tmp_path,
            prompt="hello",
            model="sonnet",
            max_budget_usd=3.0,
            system_prompt="be helpful",
        )
        backend = ClaudeBackend(DEFAULT_BACKEND_CFG)
        cmd = backend.build_cli_cmd(config)
        assert cmd[0] == "claude" or cmd[0].endswith("/claude")
        assert "-p" in cmd
        assert "hello" in cmd
        assert "--model" in cmd
        assert "sonnet" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "--system-prompt" in cmd
        assert "be helpful" in cmd
        assert "--max-budget-usd" in cmd
        assert "3.0" in cmd

    def test_build_cli_cmd_no_system_prompt(self, tmp_path: Path) -> None:
        """No --system-prompt flag when system_prompt is empty."""
        config = SandboxConfig(sandbox_dir=tmp_path, prompt="hi")
        cmd = ClaudeBackend(DEFAULT_BACKEND_CFG).build_cli_cmd(config)
        assert "--system-prompt" not in cmd

    def test_build_cli_cmd_no_budget_when_zero(self, tmp_path: Path) -> None:
        """No --max-budget-usd flag when budget is zero."""
        config = SandboxConfig(sandbox_dir=tmp_path, prompt="hi", max_budget_usd=0)
        cmd = ClaudeBackend(DEFAULT_BACKEND_CFG).build_cli_cmd(config)
        assert "--max-budget-usd" not in cmd

    def test_build_cli_cmd_persists_session_by_default(self, tmp_path: Path) -> None:
        """Sessions persist (resumable) and no --continue without a resume request."""
        config = SandboxConfig(sandbox_dir=tmp_path, prompt="hi")
        cmd = ClaudeBackend(DEFAULT_BACKEND_CFG).build_cli_cmd(config)
        assert "--no-session-persistence" not in cmd
        assert "--continue" not in cmd

    def test_build_cli_cmd_resume_adds_continue(self, tmp_path: Path) -> None:
        """resume_previous_session appends --continue to reattach to the run."""
        config = SandboxConfig(
            sandbox_dir=tmp_path, prompt="hi", resume_previous_session=True
        )
        cmd = ClaudeBackend(DEFAULT_BACKEND_CFG).build_cli_cmd(config)
        assert "--continue" in cmd
        assert "--no-session-persistence" not in cmd

    def test_build_cli_cmd_includes_tools(self, tmp_path: Path) -> None:
        """All standard tools are listed in --tools."""
        config = SandboxConfig(sandbox_dir=tmp_path, prompt="hi")
        cmd = ClaudeBackend(DEFAULT_BACKEND_CFG).build_cli_cmd(config)
        assert "--tools" in cmd
        tools_idx = cmd.index("--tools")
        tools_str = cmd[tools_idx + 1]
        for tool in ("Bash", "Read", "Write", "Edit", "Glob", "Grep", "Task"):
            assert tool in tools_str

    def test_build_env_strips_claudecode(self) -> None:
        """CLAUDECODE* env vars are removed from the child environment."""
        config = SandboxConfig(sandbox_dir=Path("/tmp/test"))
        with patch.dict(os.environ, {"CLAUDECODE_FOO": "bar"}, clear=False):
            env = ClaudeBackend(DEFAULT_BACKEND_CFG).build_env(config)
        assert "CLAUDECODE_FOO" not in env
        assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "16384"
        assert env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "80"

    def test_build_env_with_extra(self) -> None:
        """Extra env vars are merged into the environment."""
        config = SandboxConfig(sandbox_dir=Path("/tmp/test"))
        env = ClaudeBackend(DEFAULT_BACKEND_CFG).build_env(
            config, extra={"MY_KEY": "val"}
        )
        assert env["MY_KEY"] == "val"

    def test_setup_sandbox_files_creates_claude_md(self, tmp_path: Path) -> None:
        """CLAUDE.md is created with relative-path instructions."""
        config = SandboxConfig(sandbox_dir=tmp_path)
        ClaudeBackend(DEFAULT_BACKEND_CFG).setup_sandbox_files(config)
        assert (tmp_path / "CLAUDE.md").exists()
        assert "relative paths" in (tmp_path / "CLAUDE.md").read_text()

    def test_setup_sandbox_files_creates_settings(self, tmp_path: Path) -> None:
        """.claude/settings.json has the PreToolUse write-validation hook."""
        config = SandboxConfig(sandbox_dir=tmp_path)
        ClaudeBackend(DEFAULT_BACKEND_CFG).setup_sandbox_files(config)
        settings_path = tmp_path / ".claude" / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text())
        assert "hooks" in settings
        assert "PreToolUse" in settings["hooks"]

    def test_setup_sandbox_files_creates_validate_script(self, tmp_path: Path) -> None:
        """.claude/validate_sandbox.py is created."""
        config = SandboxConfig(sandbox_dir=tmp_path)
        ClaudeBackend(DEFAULT_BACKEND_CFG).setup_sandbox_files(config)
        assert (tmp_path / ".claude" / "validate_sandbox.py").exists()

    def test_setup_sandbox_files_docker_python(self, tmp_path: Path) -> None:
        """CLAUDE.md references the Docker Python path when provided."""
        config = SandboxConfig(sandbox_dir=tmp_path)
        ClaudeBackend(DEFAULT_BACKEND_CFG).setup_sandbox_files(
            config, docker_python="/custom/python"
        )
        text = (tmp_path / "CLAUDE.md").read_text()
        assert "/custom/python" in text

    def test_claude_rate_limit_regex_old_format(self) -> None:
        """Matches the old 'out of extra usage' message."""
        msg = "You are out of extra usage. Your limit resets 3am"
        m = _RATE_LIMIT_RE.search(msg)
        assert m is not None
        assert m.group(1) == "3am"

    def test_claude_rate_limit_regex_new_format(self) -> None:
        """Matches the new 'hit your limit' message."""
        msg = "You've hit your limit \u00b7 resets 2pm (Etc/Unknown)"
        m = _RATE_LIMIT_RE.search(msg)
        assert m is not None
        assert m.group(1) == "2pm"


# ---------------------------------------------------------------------------
# OpenCodeBackend
# ---------------------------------------------------------------------------


class TestOpenCodeBackend:
    """Tests for the OpenCode CLI backend."""

    def test_build_cli_cmd_basics(self, tmp_path: Path) -> None:
        """CLI command includes opencode run with --format json."""
        config = SandboxConfig(
            sandbox_dir=tmp_path,
            prompt="hello",
            model="openai/gpt-4o",
        )
        cmd = OpenCodeBackend(DEFAULT_OPENCODE_CFG).build_cli_cmd(config)
        assert cmd[0] == "opencode" or cmd[0].endswith("/opencode")
        assert "run" in cmd
        assert "hello" in cmd
        assert "--format" in cmd
        assert "json" in cmd
        assert "--model" in cmd
        assert "openai/gpt-4o" in cmd

    def test_build_cli_cmd_no_claude_flags(self, tmp_path: Path) -> None:
        """OpenCode CLI should NOT include Claude-specific flags."""
        config = SandboxConfig(
            sandbox_dir=tmp_path,
            prompt="hi",
            model="openai/gpt-4o",
            system_prompt="be helpful",
        )
        cmd = OpenCodeBackend(DEFAULT_OPENCODE_CFG).build_cli_cmd(config)
        assert "--dangerously-skip-permissions" not in cmd
        assert "--output-format" not in cmd
        assert "--system-prompt" not in cmd
        assert "--tools" not in cmd
        assert "--max-budget-usd" not in cmd

    def test_build_env_strips_vars(self) -> None:
        """CLAUDECODE* and OPENCODE* env vars are stripped."""
        config = SandboxConfig(sandbox_dir=Path("/tmp/test"))
        with patch.dict(
            os.environ,
            {"CLAUDECODE_X": "1", "OPENCODE_Y": "2"},
            clear=False,
        ):
            env = OpenCodeBackend(DEFAULT_OPENCODE_CFG).build_env(config)
        assert "CLAUDECODE_X" not in env
        assert "OPENCODE_Y" not in env
        assert env["OPENCODE_DISABLE_CLAUDE_CODE"] == "1"

    def test_build_env_with_extra(self) -> None:
        """Extra env vars are merged into the environment."""
        config = SandboxConfig(sandbox_dir=Path("/tmp/test"))
        env = OpenCodeBackend(DEFAULT_OPENCODE_CFG).build_env(
            config, extra={"OPENAI_API_KEY": "sk"}
        )
        assert env["OPENAI_API_KEY"] == "sk"

    def test_setup_sandbox_files_creates_opencode_json(self, tmp_path: Path) -> None:
        """opencode.json is created with model and permission config."""
        config = SandboxConfig(
            sandbox_dir=tmp_path,
            model="openai/gpt-4o",
        )
        OpenCodeBackend(DEFAULT_OPENCODE_CFG).setup_sandbox_files(config)
        oc_path = tmp_path / "opencode.json"
        assert oc_path.exists()
        oc = json.loads(oc_path.read_text())
        assert oc["model"] == "openai/gpt-4o"
        assert oc["permission"] == "allow"
        assert oc["compaction"]["auto"] is True

    def test_setup_sandbox_files_creates_agents_md(self, tmp_path: Path) -> None:
        """AGENTS.md includes the system prompt and relative-path rules."""
        config = SandboxConfig(
            sandbox_dir=tmp_path,
            system_prompt="You are a policy writer",
        )
        OpenCodeBackend(DEFAULT_OPENCODE_CFG).setup_sandbox_files(config)
        agents_md = tmp_path / "AGENTS.md"
        assert agents_md.exists()
        text = agents_md.read_text()
        assert "You are a policy writer" in text
        assert "relative paths" in text

    def test_setup_sandbox_files_mcp_http_uses_remote(self, tmp_path: Path) -> None:
        """Http MCP transport maps to an OpenCode "remote" server in opencode.json."""
        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        (tmp_path / "env_config.json").write_text("{}")
        config = SandboxConfig(
            sandbox_dir=sandbox_dir,
            model="openai/gpt-4o",
            mcp_tools=("render_state",),
        )
        backend = OpenCodeBackend(DEFAULT_OPENCODE_CFG)
        backend.build_cli_cmd(
            config,
            mcp_python_cmd="python",
            mcp_env_config_path=str(sandbox_dir / ".mcp" / "env_config.json"),
            mcp_transport="http",
            mcp_port=8799,
        )
        # The http transport must emit the start script the pre-start flow runs.
        assert (sandbox_dir / ".mcp" / "start_server.sh").exists()
        backend.setup_sandbox_files(config)
        oc = json.loads((sandbox_dir / "opencode.json").read_text())
        server = oc["mcp"]["robocode-tools"]
        assert server["type"] == "remote"
        assert server["url"] == "http://127.0.0.1:8799/mcp"
        assert server["enabled"] is True

    def test_setup_sandbox_files_mcp_stdio_uses_local(self, tmp_path: Path) -> None:
        """Default stdio MCP transport maps to an OpenCode "local" command server."""
        sandbox_dir = tmp_path / "sandbox"
        sandbox_dir.mkdir()
        (tmp_path / "env_config.json").write_text("{}")
        config = SandboxConfig(
            sandbox_dir=sandbox_dir,
            model="openai/gpt-4o",
            mcp_tools=("render_state",),
        )
        backend = OpenCodeBackend(DEFAULT_OPENCODE_CFG)
        backend.build_cli_cmd(
            config,
            mcp_python_cmd="python",
            mcp_env_config_path=str(sandbox_dir / ".mcp" / "env_config.json"),
        )
        assert not (sandbox_dir / ".mcp" / "start_server.sh").exists()
        backend.setup_sandbox_files(config)
        oc = json.loads((sandbox_dir / "opencode.json").read_text())
        server = oc["mcp"]["robocode-tools"]
        assert server["type"] == "local"
        assert server["command"][0] == "bash"

    def test_setup_sandbox_files_no_claude_files(self, tmp_path: Path) -> None:
        """OpenCode backend should NOT create .claude/ or CLAUDE.md."""
        config = SandboxConfig(sandbox_dir=tmp_path)
        OpenCodeBackend(DEFAULT_OPENCODE_CFG).setup_sandbox_files(config)
        assert not (tmp_path / "CLAUDE.md").exists()
        assert not (tmp_path / ".claude").exists()

    def test_setup_sandbox_files_docker_python(self, tmp_path: Path) -> None:
        """AGENTS.md references the Docker Python path when provided."""
        config = SandboxConfig(sandbox_dir=tmp_path)
        OpenCodeBackend(DEFAULT_OPENCODE_CFG).setup_sandbox_files(
            config, docker_python="/venv/bin/python"
        )
        text = (tmp_path / "AGENTS.md").read_text()
        assert "/venv/bin/python" in text

    def test_setup_sandbox_files_ollama_provider_injected(self, tmp_path: Path) -> None:
        """opencode.json gets an Ollama provider block for ollama/ models.

        Default host is the loopback (local/apptainer share the host network).
        """
        cfg = DictConfig({"backend": "opencode", "model": "ollama/qwen3.5:27b"})
        config = SandboxConfig(sandbox_dir=tmp_path, model="ollama/qwen3.5:27b")
        OpenCodeBackend(cfg).setup_sandbox_files(config)
        oc = json.loads((tmp_path / "opencode.json").read_text())
        assert "provider" in oc
        assert "ollama" in oc["provider"]
        ollama_cfg = oc["provider"]["ollama"]
        assert ollama_cfg["options"]["baseURL"] == "http://127.0.0.1:11434/v1"
        assert "qwen3.5:27b" in ollama_cfg["models"]

    def test_setup_sandbox_files_ollama_docker_host(self, tmp_path: Path) -> None:
        """The baseURL host comes from config.local_model_host (docker override)."""
        cfg = DictConfig({"backend": "opencode", "model": "ollama/qwen3:0.6b"})
        config = SandboxConfig(
            sandbox_dir=tmp_path,
            model="ollama/qwen3:0.6b",
            local_model_host="host.docker.internal",
        )
        OpenCodeBackend(cfg).setup_sandbox_files(config)
        oc = json.loads((tmp_path / "opencode.json").read_text())
        url = oc["provider"]["ollama"]["options"]["baseURL"]
        assert url == "http://host.docker.internal:11434/v1"

    def test_setup_sandbox_files_vllm_provider_injected(self, tmp_path: Path) -> None:
        """opencode.json gets a vLLM provider block (port 8000) for vllm/ models."""
        cfg = DictConfig({"backend": "opencode", "model": "vllm/Qwen/Qwen2.5-0.5B"})
        config = SandboxConfig(sandbox_dir=tmp_path, model="vllm/Qwen/Qwen2.5-0.5B")
        OpenCodeBackend(cfg).setup_sandbox_files(config)
        oc = json.loads((tmp_path / "opencode.json").read_text())
        assert "vllm" in oc["provider"]
        vllm_cfg = oc["provider"]["vllm"]
        assert vllm_cfg["options"]["baseURL"] == "http://127.0.0.1:8000/v1"
        assert "Qwen/Qwen2.5-0.5B" in vllm_cfg["models"]

    def test_setup_sandbox_files_no_provider_for_remote_model(
        self, tmp_path: Path
    ) -> None:
        """opencode.json does NOT get a provider block for remote (API) models."""
        config = SandboxConfig(sandbox_dir=tmp_path, model="openai/gpt-4o")
        OpenCodeBackend(DEFAULT_OPENCODE_CFG).setup_sandbox_files(config)
        oc = json.loads((tmp_path / "opencode.json").read_text())
        assert "provider" not in oc

    def test_build_env_calls_ensure_ollama(self) -> None:
        """build_env triggers ensure_ollama for ollama/ models."""
        cfg = DictConfig(
            {
                "backend": "opencode",
                "model": "ollama/qwen3.5:27b",
                "ollama_keep_alive": "3m",
            }
        )
        config = SandboxConfig(
            sandbox_dir=Path("/tmp/test"), model="ollama/qwen3.5:27b"
        )
        with patch("robocode.utils.backends.opencode.ensure_ollama") as mock_ensure:
            OpenCodeBackend(cfg).build_env(config)
        mock_ensure.assert_called_once_with(keep_alive="3m")

    def test_build_env_skips_ensure_ollama_for_non_ollama(self) -> None:
        """build_env does NOT call ensure_ollama for non-Ollama models."""
        config = SandboxConfig(sandbox_dir=Path("/tmp/test"), model="openai/gpt-4o")
        with patch("robocode.utils.backends.opencode.ensure_ollama") as mock_ensure:
            OpenCodeBackend(DEFAULT_OPENCODE_CFG).build_env(config)
        mock_ensure.assert_not_called()

    def test_build_env_skips_ensure_ollama_for_vllm(self) -> None:
        """build_env does NOT auto-start ollama for vLLM models (user-managed)."""
        config = SandboxConfig(
            sandbox_dir=Path("/tmp/test"), model="vllm/Qwen/Qwen2.5-0.5B"
        )
        with patch("robocode.utils.backends.opencode.ensure_ollama") as mock_ensure:
            OpenCodeBackend(DEFAULT_OPENCODE_CFG).build_env(config)
        mock_ensure.assert_not_called()


# ---------------------------------------------------------------------------
# ClaudeBackend Ollama integration
# ---------------------------------------------------------------------------


class TestClaudeOllama:
    """Tests for Claude backend Ollama integration."""

    def test_build_env_calls_ensure_ollama(self) -> None:
        """build_env triggers ensure_ollama when base_url points to Ollama."""
        cfg = DictConfig(
            {
                "backend": "claude",
                "model": "qwen3.5:27b",
                "base_url": "http://localhost:11434",
                "auth_token": "ollama",
                "ollama_keep_alive": "7m",
            }
        )
        config = SandboxConfig(sandbox_dir=Path("/tmp/test"))
        with patch("robocode.utils.backends.claude.ensure_ollama") as mock_ensure:
            ClaudeBackend(cfg).build_env(config)
        mock_ensure.assert_called_once_with(keep_alive="7m")

    def test_build_env_skips_ensure_ollama_without_base_url(self) -> None:
        """build_env does NOT call ensure_ollama for standard Claude."""
        config = SandboxConfig(sandbox_dir=Path("/tmp/test"))
        with patch("robocode.utils.backends.claude.ensure_ollama") as mock_ensure:
            ClaudeBackend(DEFAULT_BACKEND_CFG).build_env(config)
        mock_ensure.assert_not_called()

    def test_build_env_default_keep_alive(self) -> None:
        """Defaults to 5m keep_alive when not configured."""
        cfg = DictConfig(
            {
                "backend": "claude",
                "model": "qwen3.5:27b",
                "base_url": "http://localhost:11434",
                "auth_token": "ollama",
            }
        )
        config = SandboxConfig(sandbox_dir=Path("/tmp/test"))
        with patch("robocode.utils.backends.claude.ensure_ollama") as mock_ensure:
            ClaudeBackend(cfg).build_env(config)
        mock_ensure.assert_called_once_with(keep_alive="5m")


# ---------------------------------------------------------------------------
# Provider registry and utility functions
# ---------------------------------------------------------------------------


class TestProviderUtils:
    """Tests for provider registry and utility functions."""

    def test_provider_from_model_with_slash(self) -> None:
        """Extracts provider prefix before the slash."""
        assert provider_from_model("openai/gpt-4o") == "openai"
        assert provider_from_model("google/gemini-2.5-pro") == "google"
        assert provider_from_model("anthropic/claude-sonnet-4-5") == "anthropic"

    def test_provider_from_model_no_slash(self) -> None:
        """Returns empty string for model names without a slash."""
        assert provider_from_model("sonnet") == ""
        assert provider_from_model("") == ""

    def test_firewall_domains_for_known_providers(self) -> None:
        """Known providers return their API domains."""
        assert firewall_domains_for_provider("openai") == ["api.openai.com"]
        assert firewall_domains_for_provider("anthropic") == ["api.anthropic.com"]
        assert firewall_domains_for_provider("google") == [
            "generativelanguage.googleapis.com"
        ]

    def test_firewall_domains_for_unknown_provider(self) -> None:
        """Unknown providers (including local Ollama) return an empty list."""
        assert not firewall_domains_for_provider("cli")
        assert not firewall_domains_for_provider("custom")
        assert not firewall_domains_for_provider(provider_from_model("ollama/qwen3.5"))

    def test_firewall_domains_for_model_string(self) -> None:
        """Composing with provider_from_model covers 'provider/model' strings."""
        assert firewall_domains_for_provider(provider_from_model("openai/gpt-4o")) == [
            "api.openai.com"
        ]

    def test_firewall_domains_with_base_url(self) -> None:
        """A base_url adds its hostname to the provider domains."""
        assert firewall_domains_for_provider(
            "openai", "https://my-vllm.example.com:8000/v1"
        ) == ["api.openai.com", "my-vllm.example.com"]

    def test_providers_registry_has_expected_keys(self) -> None:
        """Registry contains the three core providers with valid metadata."""
        for key in ("openai", "anthropic", "google"):
            assert key in PROVIDERS
            assert PROVIDERS[key].domains
            assert PROVIDERS[key].api_key_env


# ---------------------------------------------------------------------------
# Claude stream parser
# ---------------------------------------------------------------------------


class TestClaudeParseStreamMetrics:
    """ClaudeBackend.parse_stream captures generation metrics from the stream."""

    def _make_mock_proc(self, stdout_lines: list[str]) -> subprocess.Popen:
        proc = MagicMock(spec=subprocess.Popen)
        proc.stdout = iter(stdout_lines)
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = ""
        proc.returncode = 0
        proc.wait.return_value = 0
        return proc

    def test_parse_aggregates_across_cli_sessions(self) -> None:
        """One generation can span several CLI sessions; metrics aggregate.

        Turns are counted from assistant messages (the final session's num_turns is
        stale); tokens come from the cumulative per-model usage (the top-level usage is
        empty on a budget-error session); cost and modelUsage keep the latest
        (cumulative) value; per-session fields accumulate.
        """
        events = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "planning"},
                        {"type": "tool_use", "name": "Bash", "input": {"cmd": "ls"}},
                        {"type": "tool_use", "name": "Write", "input": {"f": "a.py"}},
                    ]
                },
            },
            {
                "type": "system",
                "subtype": "compact_boundary",
                "compact_metadata": {"trigger": "auto", "pre_tokens": 1000},
            },
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "num_turns": 8,
                "total_cost_usd": 0.30,
                "duration_ms": 1000,
                "duration_api_ms": 2000,
                "permission_denials": [{"tool": "Bash"}],
                "modelUsage": {"claude-sonnet-5": {"inputTokens": 100}},
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "x"}]},
            },
            {
                "type": "result",
                "subtype": "error_max_budget_usd",
                "is_error": True,
                "num_turns": 1,
                "total_cost_usd": 0.90,
                "duration_ms": 500,
                "duration_api_ms": 5000,
                "permission_denials": [{"tool": "Write"}, {"tool": "Edit"}],
                "modelUsage": {
                    "claude-sonnet-5": {
                        "inputTokens": 300,
                        "outputTokens": 50,
                        "cacheReadInputTokens": 2000,
                        "cacheCreationInputTokens": 10,
                    },
                    "claude-haiku-4-5": {"inputTokens": 20},
                },
                "usage": {},
            },
        ]
        proc = self._make_mock_proc([json.dumps(e) + "\n" for e in events])
        result = ClaudeBackend(DEFAULT_BACKEND_CFG).parse_stream(proc)

        assert result.num_turns == 2  # two assistant messages, not the stale 1
        assert result.num_tool_calls == 2
        assert result.num_autocompactions == 1
        assert result.num_permission_denials == 3  # 1 + 2 across sessions
        assert result.input_tokens == 320  # 300 + 20, from the cumulative modelUsage
        assert result.output_tokens == 50
        assert result.cache_read_tokens == 2000
        assert result.cache_creation_tokens == 10
        assert result.cli_duration_ms == 1500  # summed per-session
        assert result.cli_duration_api_ms == 5000  # max (cumulative)
        assert result.stop_reason == "error_max_budget_usd"  # last subtype
        assert result.total_cost == pytest.approx(0.90)  # latest (cumulative)


# ---------------------------------------------------------------------------
# OpenCode stream parser
# ---------------------------------------------------------------------------


class TestOpenCodeParseStream:
    """Tests for OpenCodeBackend.parse_stream with mock processes."""

    def _make_mock_proc(
        self,
        stdout_lines: list[str],
        returncode: int = 0,
    ) -> subprocess.Popen:
        """Create a mock Popen with stdout lines (OpenCode JSON events)."""
        proc = MagicMock(spec=subprocess.Popen)
        proc.stdout = iter(stdout_lines)
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = ""
        proc.returncode = returncode
        proc.wait.return_value = returncode
        return proc

    def test_parse_text_event(self) -> None:
        """Text events are parsed and cost is accumulated."""
        events = [
            json.dumps(
                {
                    "type": "step_start",
                    "timestamp": 1,
                    "sessionID": "s1",
                    "part": {"type": "step-start"},
                }
            ),
            json.dumps(
                {
                    "type": "text",
                    "timestamp": 2,
                    "sessionID": "s1",
                    "part": {
                        "type": "text",
                        "text": "Hello world",
                        "time": {"start": 1, "end": 2},
                    },
                }
            ),
            json.dumps(
                {
                    "type": "step_finish",
                    "timestamp": 3,
                    "sessionID": "s1",
                    "part": {
                        "type": "step-finish",
                        "reason": "stop",
                        "cost": 0.001,
                        "tokens": {
                            "input": 100,
                            "output": 50,
                            "reasoning": 0,
                            "cache": {"read": 0, "write": 0},
                        },
                    },
                }
            ),
        ]
        proc = self._make_mock_proc([e + "\n" for e in events])
        result = OpenCodeBackend(DEFAULT_OPENCODE_CFG).parse_stream(proc)
        assert not result.is_error
        assert result.num_turns == 1
        assert result.total_cost == pytest.approx(0.001)

    def test_parse_error_event(self) -> None:
        """Error events set is_error and error_text."""
        events = [
            json.dumps(
                {
                    "type": "error",
                    "timestamp": 1,
                    "sessionID": "s1",
                    "error": {
                        "name": "ProviderAuthError",
                        "data": {"message": "Invalid API key"},
                    },
                }
            ),
        ]
        proc = self._make_mock_proc([e + "\n" for e in events])
        result = OpenCodeBackend(DEFAULT_OPENCODE_CFG).parse_stream(proc)
        assert result.is_error
        assert result.error_text is not None
        assert "ProviderAuthError" in result.error_text
        assert "Invalid API key" in result.error_text

    def test_parse_tool_use_event(self) -> None:
        """Tool use events are parsed without error."""
        events = [
            json.dumps(
                {
                    "type": "step_start",
                    "timestamp": 1,
                    "sessionID": "s1",
                    "part": {"type": "step-start"},
                }
            ),
            json.dumps(
                {
                    "type": "tool_use",
                    "timestamp": 2,
                    "sessionID": "s1",
                    "part": {
                        "type": "tool",
                        "tool": "bash",
                        "state": {
                            "status": "completed",
                            "input": {"command": "echo hi"},
                            "output": "hi\n",
                            "time": {"start": 1, "end": 2},
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "type": "step_finish",
                    "timestamp": 3,
                    "sessionID": "s1",
                    "part": {
                        "type": "step-finish",
                        "reason": "stop",
                        "cost": 0,
                        "tokens": {
                            "input": 0,
                            "output": 0,
                            "reasoning": 0,
                            "cache": {"read": 0, "write": 0},
                        },
                    },
                }
            ),
        ]
        proc = self._make_mock_proc([e + "\n" for e in events])
        result = OpenCodeBackend(DEFAULT_OPENCODE_CFG).parse_stream(proc)
        assert not result.is_error
        assert result.num_turns == 1

    def test_parse_nonzero_exit(self) -> None:
        """Non-zero exit code is reported as an error."""
        proc = self._make_mock_proc([], returncode=1)
        result = OpenCodeBackend(DEFAULT_OPENCODE_CFG).parse_stream(proc)
        assert result.is_error
        assert result.error_text is not None
        assert "exited with code 1" in result.error_text

    def test_parse_multiple_steps_accumulate_cost(self) -> None:
        """Cost is summed across multiple step_finish events."""
        events = []
        for i in range(3):
            events.append(
                json.dumps(
                    {
                        "type": "step_start",
                        "timestamp": i * 10,
                        "sessionID": "s1",
                        "part": {"type": "step-start"},
                    }
                )
            )
            events.append(
                json.dumps(
                    {
                        "type": "step_finish",
                        "timestamp": i * 10 + 5,
                        "sessionID": "s1",
                        "part": {
                            "type": "step-finish",
                            "reason": "stop",
                            "cost": 0.01,
                            "tokens": {
                                "input": 0,
                                "output": 0,
                                "reasoning": 0,
                                "cache": {"read": 0, "write": 0},
                            },
                        },
                    }
                )
            )
        proc = self._make_mock_proc([e + "\n" for e in events])
        result = OpenCodeBackend(DEFAULT_OPENCODE_CFG).parse_stream(proc)
        assert result.num_turns == 3
        assert result.total_cost == pytest.approx(0.03)

    def test_parse_non_json_lines_ignored(self) -> None:
        """Non-JSON lines in the stream are silently skipped."""
        lines = [
            "Not JSON at all\n",
            json.dumps(
                {
                    "type": "step_start",
                    "timestamp": 1,
                    "sessionID": "s1",
                    "part": {"type": "step-start"},
                }
            )
            + "\n",
            "Another non-JSON line\n",
            json.dumps(
                {
                    "type": "step_finish",
                    "timestamp": 2,
                    "sessionID": "s1",
                    "part": {
                        "type": "step-finish",
                        "reason": "stop",
                        "cost": 0,
                        "tokens": {
                            "input": 0,
                            "output": 0,
                            "reasoning": 0,
                            "cache": {"read": 0, "write": 0},
                        },
                    },
                }
            )
            + "\n",
        ]
        proc = self._make_mock_proc(lines)
        result = OpenCodeBackend(DEFAULT_OPENCODE_CFG).parse_stream(proc)
        assert not result.is_error
        assert result.num_turns == 1
