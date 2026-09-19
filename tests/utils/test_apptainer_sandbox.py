"""Tests for apptainer_sandbox.py.

Unit-level coverage only: verifies config defaults, that
:func:`_setup_sandbox_dir` is reused (sanity), and that
:func:`_build_apptainer_cmd` assembles the expected ``apptainer exec``
command line. No SIF or apptainer binary is invoked.
"""

import asyncio
import json
import subprocess
from pathlib import Path

import pytest
from omegaconf import DictConfig

from robocode.mcp import MCP_START_SCRIPT
from robocode.utils.apptainer_sandbox import (
    APPTAINER_PYTHON,
    ApptainerSandboxConfig,
    _build_apptainer_cmd,
    _isolated_transport,
    run_agent_in_apptainer_sandbox,
    sif_path_for,
)
from robocode.utils.backends import create_backend
from robocode.utils.docker_sandbox import (
    DOCKER_PYTHON,
    _find_repo_root,
)
from robocode.utils.isolated_transport import UnixRelay
from robocode.utils.model_broker import BrokerUpstream
from robocode.utils.strict_blackbox import (
    STRICT_BLACKBOX_PYTHON,
)


def test_apptainer_python_matches_docker_python() -> None:
    """The interpreter path inside the container is the same for both backends."""
    assert APPTAINER_PYTHON == DOCKER_PYTHON


def test_config_defaults() -> None:
    """ApptainerSandboxConfig fields have the expected defaults."""
    config = ApptainerSandboxConfig(sandbox_dir=Path("/tmp/test"))
    assert config.sif_path == _find_repo_root() / "robocode-sandbox.sif"
    assert config.model == "sonnet"
    assert config.max_budget_usd == 20.0
    assert config.system_prompt == ""
    assert config.prompt == ""
    assert config.output_filename == ""
    assert not config.init_files
    assert not config.primitive_names
    assert not config.mcp_tools


def test_config_strict_defaults() -> None:
    """Strict runs are off by default and select the strict SIF when on."""
    config = ApptainerSandboxConfig(sandbox_dir=Path("/tmp/test"))
    assert not config.blackbox_strict
    assert config.strict_sif_path == _find_repo_root() / "robocode-strict-blackbox.sif"
    assert sif_path_for(config) == config.sif_path
    strict = ApptainerSandboxConfig(
        sandbox_dir=Path("/tmp/test"), blackbox=True, blackbox_strict=True
    )
    assert sif_path_for(strict) == strict.strict_sif_path


def test_build_cmd_strict_has_no_project_mounts(tmp_path: Path) -> None:
    """A strict launch runs the strict SIF with the sandbox as its only mount."""
    config = ApptainerSandboxConfig(
        sandbox_dir=tmp_path / "sandbox",
        sif_path=tmp_path / "robocode-sandbox.sif",
        strict_sif_path=tmp_path / "robocode-strict-blackbox.sif",
        blackbox=True,
        blackbox_strict=True,
    )
    cmd = _build_apptainer_cmd(
        config,
        sandbox_abs="/host/sandbox",
        src_abs=None,
        kindergarden_abs=None,
        kinder_baselines_abs=None,
        agent_cmd=["claude"],
    )
    joined = " ".join(cmd)
    assert str(config.strict_sif_path) in cmd
    assert str(config.sif_path) not in cmd
    assert "/host/sandbox:/sandbox" in cmd
    assert "--containall" in cmd
    assert not any("ROBOCODE_SKIP_FIREWALL" in arg for arg in cmd)
    assert "/robocode/src" not in joined
    assert "kindergarden" not in joined
    assert "ss-pybullet" not in joined
    assert "ROBOCODE_UV_EXTRA_ARGS" not in joined


class _Launched(Exception):
    """Raised by the fake launcher once the command line has been captured."""


def test_strict_run_uses_only_clean_interpreter(  # type: ignore
    tmp_path: Path, monkeypatch
) -> None:
    """Agent scripts and render tools share only the strict numerical dependencies."""
    strict_sif_path = tmp_path / "robocode-strict-blackbox.sif"
    strict_sif_path.touch()
    sandbox_dir = tmp_path / "run" / "sandbox"
    metadata_path = tmp_path / "env_spaces.json"
    metadata_path.write_text(
        json.dumps({"host": "attacker.invalid", "port": 9999}), encoding="utf-8"
    )
    config = ApptainerSandboxConfig(
        sandbox_dir=sandbox_dir,
        sif_path=tmp_path / "robocode-sandbox.sif",
        strict_sif_path=strict_sif_path,
        blackbox=True,
        blackbox_strict=True,
        mcp_tools=("render_state",),
        prompt="hello",
        output_filename="approach.py",
        env_server_port=12345,
        init_files={"env_spaces.json": metadata_path},
    )
    monkeypatch.setattr(
        "robocode.utils.apptainer_sandbox.load_broker_upstream",
        lambda _: BrokerUpstream("messages", "api.anthropic.com", "/v1", {}),
    )
    targets: list[tuple[str, int]] = []

    def capture_relay(path: str, target: tuple[str, int]) -> UnixRelay:
        targets.append(target)
        return UnixRelay(path, target)

    monkeypatch.setattr("robocode.utils.apptainer_sandbox.UnixRelay", capture_relay)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "host-only-test-secret")
    launched: list[list[str]] = []
    real_popen = subprocess.Popen

    def fake_popen(cmd: list[str], **kwargs):  # type: ignore
        if cmd[0] != "apptainer":  # the sandbox's own git commands
            return real_popen(cmd, **kwargs)
        assert kwargs["start_new_session"] is True
        assert "host-only-test-secret" not in str(kwargs)
        assert ".credentials.json" not in " ".join(cmd)
        launched.append(cmd)
        raise _Launched

    monkeypatch.setattr("robocode.utils.apptainer_sandbox.subprocess.Popen", fake_popen)
    backend = create_backend(DictConfig({"backend": "claude", "model": "sonnet"}))
    with pytest.raises(_Launched):
        asyncio.run(run_agent_in_apptainer_sandbox(config, backend))

    assert targets == [("127.0.0.1", 12345)]
    metadata = json.loads((sandbox_dir / "env_spaces.json").read_text(encoding="utf-8"))
    assert (metadata["host"], metadata["port"]) == ("127.0.0.1", 18081)
    assert len(launched) == 1
    cmd = launched[0]
    joined = " ".join(cmd)
    assert str(strict_sif_path) in cmd
    assert "/robocode/src" not in joined
    # Agent scripts, the render server, and the startup probe use the same
    # dependency-clean interpreter.
    assert f"{STRICT_BLACKBOX_PYTHON} -c" in joined
    assert STRICT_BLACKBOX_PYTHON in (sandbox_dir / "CLAUDE.md").read_text()
    start_script = (sandbox_dir / ".mcp" / MCP_START_SCRIPT).read_text()
    assert (
        f"{STRICT_BLACKBOX_PYTHON} /opt/robocode-render/strict_server.py"
        in start_script
    )
    assert APPTAINER_PYTHON not in start_script


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
        agent_cmd=["claude", "--print", "hello"],
    )

    assert cmd[0] == "apptainer"
    assert cmd[1] == "exec"
    assert "--containall" in cmd
    # Own PID namespace: apptainer shares the host's by default, so without this a
    # `pkill -f` inside the sandbox reaches the harness and concurrent runs.
    assert "--pid" in cmd
    assert "--writable-tmpfs" in cmd
    assert "--fakeroot" not in cmd
    assert "--no-home" in cmd
    assert "--cleanenv" in cmd
    pwd_idx = cmd.index("--pwd")
    assert cmd[pwd_idx + 1] == "/sandbox"

    # Env vars are passed as `--env KEY=val` pairs.
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192" in cmd
    assert "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=70" in cmd
    # Apptainer uses the isolated namespace, not the Docker firewall entrypoint.
    assert not any("ROBOCODE_SKIP_FIREWALL" in arg for arg in cmd)
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
    entrypoint_idx = cmd.index("/usr/bin/setpriv")
    assert "--net" in cmd
    assert cmd[cmd.index("--network") + 1] == "none"
    assert "--userns" in cmd
    assert "/usr/local/bin/entrypoint.sh" not in cmd
    assert sif_idx < entrypoint_idx

    # Agent command is appended at the end.
    assert cmd[-3:] == ["claude", "--print", "hello"]


def test_build_cmd_bilevel_conditional(tmp_path: Path) -> None:
    """Bilevel source is conditional; dependency installation is a separate phase."""

    def build(kinder_baselines_abs: str | None) -> list[str]:
        return _build_apptainer_cmd(
            ApptainerSandboxConfig(sandbox_dir=tmp_path / "sandbox"),
            sandbox_abs="/host/sandbox",
            src_abs="/host/src",
            kindergarden_abs="/host/kindergarden",
            kinder_baselines_abs=kinder_baselines_abs,
            agent_cmd=["claude"],
        )

    off = build(None)
    assert "kinder-baselines" not in " ".join(off)
    assert "ROBOCODE_UV_EXTRA_ARGS=--extra bilevel" not in off

    on = build("/host/kinder-baselines")
    assert "/host/kinder-baselines:/robocode/third-party/kinder-baselines" in on
    assert not any("ROBOCODE_UV_EXTRA_ARGS" in arg for arg in on)


def test_build_cmd_always_adds_containall(tmp_path: Path) -> None:
    """Every mode adds --containall; --no-home alone can leak host paths.

    Without --containall, many apptainer.conf setups still bind the host /home, so an
    agent could read experimenter-side Hydra configuration as well as the real
    environment source. --containall drops all default binds so only the filtered mounts
    remain.
    """
    blackbox_cmd = _build_apptainer_cmd(
        ApptainerSandboxConfig(sandbox_dir=tmp_path / "sandbox", blackbox=True),
        sandbox_abs="/host/sandbox",
        src_abs="/host/src",
        kindergarden_abs="/host/kindergarden",
        kinder_baselines_abs=None,
        agent_cmd=["claude"],
    )
    default_cmd = _build_apptainer_cmd(
        ApptainerSandboxConfig(sandbox_dir=tmp_path / "sandbox"),
        sandbox_abs="/host/sandbox",
        src_abs="/host/src",
        kindergarden_abs="/host/kindergarden",
        kinder_baselines_abs=None,
        agent_cmd=["claude"],
    )
    assert "--containall" in blackbox_cmd
    assert "--containall" in default_cmd
    # --containall implies --pid, and it remains explicit in both modes.
    assert "--pid" in blackbox_cmd
    assert "--pid" in default_cmd


def test_metadata_cannot_select_an_environment_destination(tmp_path: Path) -> None:
    """Only the explicit host config can authorize a relay, including on resume."""
    (tmp_path / "env_spaces.json").write_text(
        json.dumps({"host": "127.0.0.1", "port": 9999}), encoding="utf-8"
    )
    config = ApptainerSandboxConfig(sandbox_dir=tmp_path)
    upstream = BrokerUpstream("messages", "api.anthropic.com", "/v1", {})
    with pytest.raises(RuntimeError, match="explicit trusted env_server_port"):
        with _isolated_transport(config, upstream):
            pytest.fail("Untrusted metadata enabled a host relay")


def test_unsupported_backend_never_sets_up_an_agent(tmp_path, monkeypatch) -> None:
    """Removing the old OpenCode auth path cannot cause an unbrokered fallback."""
    image = tmp_path / "image.sif"
    image.touch()
    config = ApptainerSandboxConfig(sandbox_dir=tmp_path / "sandbox", sif_path=image)
    backend = create_backend(
        DictConfig({"backend": "opencode", "model": "openai/test"})
    )
    monkeypatch.setattr(
        "robocode.utils.apptainer_sandbox._setup_sandbox_dir",
        lambda _: pytest.fail("Unsupported backend reached agent setup"),
    )
    with pytest.raises(RuntimeError, match="does not support 'opencode'"):
        asyncio.run(run_agent_in_apptainer_sandbox(config, backend))
