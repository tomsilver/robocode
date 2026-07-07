"""Tests for apptainer_sandbox.py.

Unit-level coverage only: verifies config defaults, that
:func:`_setup_sandbox_dir` is reused (sanity), and that
:func:`_build_apptainer_cmd` assembles the expected ``apptainer exec``
command line. No SIF or apptainer binary is invoked.
"""

from pathlib import Path

from robocode.utils.apptainer_sandbox import (
    APPTAINER_PYTHON,
    ApptainerSandboxConfig,
    _build_apptainer_auth_args,
    _build_apptainer_cmd,
)
from robocode.utils.docker_sandbox import DOCKER_PYTHON, _find_repo_root


def test_apptainer_python_matches_docker_python() -> None:
    """The interpreter path inside the container is the same for both backends."""
    assert APPTAINER_PYTHON == DOCKER_PYTHON


def test_config_defaults() -> None:
    """ApptainerSandboxConfig fields have the expected defaults."""
    config = ApptainerSandboxConfig(sandbox_dir=Path("/tmp/test"))
    assert config.sif_path == _find_repo_root() / "robocode-sandbox.sif"
    assert config.model == "sonnet"
    assert config.max_budget_usd == 5.0
    assert config.system_prompt == ""
    assert config.prompt == ""
    assert config.output_filename == ""
    assert not config.init_files
    assert not config.primitive_names
    assert not config.mcp_tools


def test_build_cmd_basic_shape(tmp_path: Path) -> None:
    """_build_apptainer_cmd produces the expected flag layout."""
    config = ApptainerSandboxConfig(
        sandbox_dir=tmp_path / "sandbox",
        sif_path=tmp_path / "robocode-sandbox.sif",
        max_output_tokens=8192,
        autocompact_pct=70,
    )
    cmd = _build_apptainer_cmd(
        config,
        sandbox_abs="/host/sandbox",
        src_abs="/host/src",
        kindergarden_abs="/host/kindergarden",
        kinder_baselines_abs=None,
        auth_args=[],
        firewall_domains=[],
        agent_cmd=["claude", "--print", "hello"],
    )

    assert cmd[0] == "apptainer"
    assert cmd[1] == "exec"
    assert "--writable-tmpfs" in cmd
    assert "--fakeroot" not in cmd
    assert "--no-home" in cmd
    assert "--cleanenv" in cmd
    pwd_idx = cmd.index("--pwd")
    assert cmd[pwd_idx + 1] == "/sandbox"

    # Env vars are passed as `--env KEY=val` pairs.
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192" in cmd
    assert "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=70" in cmd
    # init-firewall.sh is skipped (apptainer can't grant CAP_NET_ADMIN).
    assert "ROBOCODE_SKIP_FIREWALL=1" in cmd
    # Headless container has no GPU: mujoco's Dynamic3D renderer must use OSMesa
    # (software), so the sandbox forces it; EGL device displays would crash.
    assert "MUJOCO_GL=osmesa" in cmd
    assert "PYOPENGL_PLATFORM=osmesa" in cmd

    # Bind mounts.
    assert "/host/sandbox:/sandbox" in cmd
    assert "/host/src:/robocode/src" in cmd
    assert "/host/kindergarden:/robocode/third-party/kindergarden" in cmd

    # SIF path appears before the entrypoint invocation.
    sif_idx = cmd.index(str(config.sif_path))
    entrypoint_idx = cmd.index("/usr/local/bin/entrypoint.sh")
    assert sif_idx < entrypoint_idx

    # Agent command is appended at the end.
    assert cmd[-3:] == ["claude", "--print", "hello"]


def test_build_cmd_bilevel_conditional(tmp_path: Path) -> None:
    """The kinder-baselines bind and --extra bilevel sync appear only when requested."""

    def build(kinder_baselines_abs: str | None) -> list[str]:
        return _build_apptainer_cmd(
            ApptainerSandboxConfig(sandbox_dir=tmp_path / "sandbox"),
            sandbox_abs="/host/sandbox",
            src_abs="/host/src",
            kindergarden_abs="/host/kindergarden",
            kinder_baselines_abs=kinder_baselines_abs,
            auth_args=[],
            firewall_domains=[],
            agent_cmd=["claude"],
        )

    off = build(None)
    assert "kinder-baselines" not in " ".join(off)
    assert "ROBOCODE_UV_EXTRA_ARGS=--extra bilevel" not in off

    on = build("/host/kinder-baselines")
    assert "/host/kinder-baselines:/robocode/third-party/kinder-baselines" in on
    assert "ROBOCODE_UV_EXTRA_ARGS=--extra bilevel" in on


def test_build_cmd_blackbox_adds_containall(tmp_path: Path) -> None:
    """Blackbox mode adds --containall; --no-home alone leaks the host /home.

    Without --containall, many apptainer.conf setups still bind the host /home,
    so a blackbox agent could read the real env source at
    /home/<user>/.../src/robocode/environments. --containall drops all default
    binds so only the filtered mounts remain. The default (non-blackbox) command
    keeps its existing flags.
    """
    blackbox_cmd = _build_apptainer_cmd(
        ApptainerSandboxConfig(sandbox_dir=tmp_path / "sandbox", blackbox=True),
        sandbox_abs="/host/sandbox",
        src_abs="/host/src",
        kindergarden_abs="/host/kindergarden",
        kinder_baselines_abs=None,
        auth_args=[],
        firewall_domains=[],
        agent_cmd=["claude"],
    )
    default_cmd = _build_apptainer_cmd(
        ApptainerSandboxConfig(sandbox_dir=tmp_path / "sandbox"),
        sandbox_abs="/host/sandbox",
        src_abs="/host/src",
        kindergarden_abs="/host/kindergarden",
        kinder_baselines_abs=None,
        auth_args=[],
        firewall_domains=[],
        agent_cmd=["claude"],
    )
    assert "--containall" in blackbox_cmd
    assert "--containall" not in default_cmd


def test_build_cmd_firewall_domains(tmp_path: Path) -> None:
    """Firewall domains, when present, are forwarded via --env."""
    config = ApptainerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    cmd = _build_apptainer_cmd(
        config,
        sandbox_abs="/host/sandbox",
        src_abs="/host/src",
        kindergarden_abs="/host/kindergarden",
        kinder_baselines_abs=None,
        auth_args=[],
        firewall_domains=["api.example.com", "cdn.example.com"],
        agent_cmd=["claude"],
    )
    assert "ROBOCODE_FIREWALL_EXTRA_DOMAINS=api.example.com,cdn.example.com" in cmd


def test_build_cmd_no_firewall_when_empty(tmp_path: Path) -> None:
    """When no extra domains are requested, the env var is not added."""
    config = ApptainerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    cmd = _build_apptainer_cmd(
        config,
        sandbox_abs="/host/sandbox",
        src_abs="/host/src",
        kindergarden_abs="/host/kindergarden",
        kinder_baselines_abs=None,
        auth_args=[],
        firewall_domains=[],
        agent_cmd=["claude"],
    )
    assert not any("ROBOCODE_FIREWALL_EXTRA_DOMAINS" in arg for arg in cmd)


def test_build_cmd_auth_args_inserted(tmp_path: Path) -> None:
    """Caller-supplied auth args (e.g. a --bind) appear in the cmd."""
    config = ApptainerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    auth_args = ["--bind", "/home/u/.claude:/home/node/.claude"]
    cmd = _build_apptainer_cmd(
        config,
        sandbox_abs="/host/sandbox",
        src_abs="/host/src",
        kindergarden_abs="/host/kindergarden",
        kinder_baselines_abs=None,
        auth_args=auth_args,
        firewall_domains=[],
        agent_cmd=["claude"],
    )
    assert "/home/u/.claude:/home/node/.claude" in cmd


def test_opencode_auth_passes_api_keys(monkeypatch) -> None:  # type: ignore
    """Provider API keys are forwarded via APPTAINERENV_ env vars, not argv."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-value")
    args, env = _build_apptainer_auth_args("opencode")
    assert env.get("APPTAINERENV_ANTHROPIC_API_KEY") == "sk-test-value"
    # The secret must not appear on the command line.
    assert not any("sk-test-value" in a for a in args)


def test_claude_auth_uses_env_token(monkeypatch) -> None:  # type: ignore
    """CLAUDE_CODE_OAUTH_TOKEN is forwarded via APPTAINERENV_, never on argv."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
    args, env = _build_apptainer_auth_args("claude")
    assert env.get("APPTAINERENV_CLAUDE_CODE_OAUTH_TOKEN") == "sk-ant-oat01-test"
    # The token must not appear on the command line (visible via `ps`).
    assert not any("sk-ant-oat01-test" in a for a in args)
    assert not any("--bind" in a for a in args)


def test_claude_auth_falls_back_to_bind(monkeypatch) -> None:  # type: ignore
    """When no token is found, falls back to bind-mounting ~/.claude."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    # Force the resolver to report no token (avoid Keychain hit on dev macOS).
    monkeypatch.setattr(
        "robocode.utils.apptainer_sandbox._get_claude_oauth_token",
        lambda: None,
    )
    args, env = _build_apptainer_auth_args("claude")
    assert not env  # no env vars when bind is used
    assert "--bind" in args
    assert any(arg.endswith(":/home/node/.claude") for arg in args)
