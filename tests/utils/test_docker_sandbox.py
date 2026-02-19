"""Tests for docker_sandbox.py.

Tests are split into two categories:

**Unit tests** (no Docker required)
    Verify config defaults, repo-root detection, and that
    :func:`_setup_sandbox_dir` creates the correct files on disk.

**Integration tests** (require the ``robocode-sandbox`` Docker image)
    Spin up the container with ``--entrypoint /bin/bash`` (bypassing the
    firewall init so no ``NET_ADMIN`` capability is needed) and assert that
    the agent's view matches expectations:

    * ``/sandbox/`` contains exactly the expected top-level entries.
    * ``/sandbox/primitives/`` holds the allowed primitive ``.py`` files.
    * ``/robocode/.venv/bin/python`` is executable and reports Python 3.11.
    * Key packages (numpy, gymnasium, robocode) are importable via the venv.
    * ``/robocode/prpl-mono/`` reflects the bind-mounted submodule.
    * prpl-mono packages (e.g. ``relational_structs``) are importable.
    * Robocode source is **not** present directly inside ``/sandbox/``.
    * The ``validate_sandbox.py`` hook blocks writes outside ``/sandbox``.

Integration tests are skipped automatically when the image is unavailable;
build it with ``bash docker/build.sh`` to run them.
"""

import subprocess
from pathlib import Path

import pytest

from robocode.utils.docker_sandbox import (
    DOCKER_PYTHON,
    DockerSandboxConfig,
    _PRIMITIVES_SRC,
    _find_repo_root,
    _setup_sandbox_dir,
)

_DOCKER_IMAGE = "robocode-sandbox"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_available() -> bool:
    """Return True if the robocode-sandbox Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", _DOCKER_IMAGE],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


requires_docker = pytest.mark.skipif(
    not _image_available(),
    reason=(
        f"Docker image '{_DOCKER_IMAGE}' not available; "
        "build it with: bash docker/build.sh"
    ),
)


def _run_in_container(
    sandbox_dir: Path, bash_cmd: str
) -> subprocess.CompletedProcess[str]:
    """Run *bash_cmd* inside the container with the sandbox and prpl-mono mounted.

    Uses ``--entrypoint /bin/bash`` to skip ``init-firewall.sh`` so that
    no ``NET_ADMIN`` capability is required during tests.
    """
    repo_root = _find_repo_root()
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/bin/bash",
            "-v",
            f"{sandbox_dir.resolve()}:/sandbox",
            "-v",
            f"{(repo_root / 'prpl-mono').resolve()}:/robocode/prpl-mono:ro",
            "-w",
            "/sandbox",
            _DOCKER_IMAGE,
            "-c",
            bash_cmd,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# Unit tests — no Docker needed
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    """DockerSandboxConfig fields have the expected defaults."""
    config = DockerSandboxConfig(sandbox_dir=Path("/tmp/test"))
    assert config.docker_image == _DOCKER_IMAGE
    assert config.model == "sonnet"
    assert config.max_budget_usd == 5.0
    assert config.system_prompt == ""
    assert config.prompt == ""
    assert config.output_filename == ""
    assert config.init_files == {}


def test_find_repo_root_has_pyproject() -> None:
    """_find_repo_root() returns a directory containing pyproject.toml."""
    root = _find_repo_root()
    assert (root / "pyproject.toml").exists()


def test_find_repo_root_has_prpl_mono() -> None:
    """_find_repo_root() returns a directory containing prpl-mono/."""
    root = _find_repo_root()
    assert (root / "prpl-mono").is_dir()


def test_setup_creates_sandbox_dir(tmp_path: Path) -> None:
    """_setup_sandbox_dir() creates the sandbox directory if absent."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    assert not config.sandbox_dir.exists()
    _setup_sandbox_dir(config)
    assert config.sandbox_dir.is_dir()


def test_setup_creates_claude_md(tmp_path: Path) -> None:
    """CLAUDE.md is created and references the Docker Python path."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    claude_md = config.sandbox_dir / "CLAUDE.md"
    assert claude_md.exists()
    assert DOCKER_PYTHON in claude_md.read_text()


def test_setup_creates_dot_claude_settings(tmp_path: Path) -> None:
    """.claude/settings.json is created with the PreToolUse hook."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    settings = config.sandbox_dir / ".claude" / "settings.json"
    assert settings.exists()
    assert "PreToolUse" in settings.read_text()


def test_setup_creates_validate_script(tmp_path: Path) -> None:
    """.claude/validate_sandbox.py is created."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    assert (config.sandbox_dir / ".claude" / "validate_sandbox.py").exists()


def test_setup_creates_git_repo(tmp_path: Path) -> None:
    """_setup_sandbox_dir() initialises a git repo inside the sandbox."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    assert (config.sandbox_dir / ".git").is_dir()


def test_setup_copies_primitives(tmp_path: Path) -> None:
    """Primitive .py files are copied into sandbox/primitives/."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    primitives_dir = config.sandbox_dir / "primitives"
    assert primitives_dir.is_dir()
    copied = {f.name for f in primitives_dir.glob("*.py")}
    expected = {
        f.name for f in _PRIMITIVES_SRC.glob("*.py") if f.name != "__init__.py"
    }
    assert copied == expected


def test_setup_no_dunder_init_in_primitives(tmp_path: Path) -> None:
    """__init__.py is excluded from the primitives copy."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    assert not (config.sandbox_dir / "primitives" / "__init__.py").exists()


def test_setup_copies_init_files(tmp_path: Path) -> None:
    """Caller-supplied init_files are copied into the sandbox."""
    source = tmp_path / "hello.txt"
    source.write_text("hello world")
    config = DockerSandboxConfig(
        sandbox_dir=tmp_path / "sandbox",
        init_files={"subdir/hello.txt": source},
    )
    _setup_sandbox_dir(config)
    dest = config.sandbox_dir / "subdir" / "hello.txt"
    assert dest.exists()
    assert dest.read_text() == "hello world"


def test_setup_is_idempotent(tmp_path: Path) -> None:
    """Calling _setup_sandbox_dir twice raises no errors."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    _setup_sandbox_dir(config)


# ---------------------------------------------------------------------------
# Integration tests — require the robocode-sandbox Docker image
# ---------------------------------------------------------------------------


@requires_docker
def test_container_cwd_is_sandbox(tmp_path: Path) -> None:
    """The container's working directory is /sandbox."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(config.sandbox_dir, "pwd")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/sandbox"


@requires_docker
def test_container_sandbox_top_level_entries(tmp_path: Path) -> None:
    """The sandbox contains exactly the expected top-level entries and no extras.

    Expected entries: primitives/, CLAUDE.md, .claude/, .git/
    Must NOT contain: src/, pyproject.toml, prpl-mono/ (host repo files)
    """
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(config.sandbox_dir, "ls -a /sandbox")
    assert result.returncode == 0, result.stderr
    listed = set(result.stdout.split())

    # Expected entries.
    assert "primitives" in listed
    assert "CLAUDE.md" in listed
    assert ".claude" in listed
    assert ".git" in listed

    # Host repo files must NOT bleed into the sandbox.
    assert "pyproject.toml" not in listed
    assert "src" not in listed
    assert "prpl-mono" not in listed


@requires_docker
def test_container_primitives_files(tmp_path: Path) -> None:
    """The primitive .py files inside /sandbox/primitives/ match the source."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(config.sandbox_dir, "ls /sandbox/primitives/")
    assert result.returncode == 0, result.stderr
    listed = set(result.stdout.split())
    expected = {
        f.name for f in _PRIMITIVES_SRC.glob("*.py") if f.name != "__init__.py"
    }
    assert expected == listed


@requires_docker
def test_container_venv_python_exists(tmp_path: Path) -> None:
    """The venv Python interpreter exists and is executable."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(config.sandbox_dir, f"{DOCKER_PYTHON} --version")
    assert result.returncode == 0, result.stderr
    version_output = result.stdout + result.stderr  # Python prints to stderr on 3.x
    assert "Python 3.11" in version_output


@requires_docker
def test_container_venv_imports_core_packages(tmp_path: Path) -> None:
    """numpy, gymnasium, and robocode are importable via the venv Python."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(
        config.sandbox_dir,
        f"{DOCKER_PYTHON} -c 'import numpy, gymnasium, robocode; print(\"OK\")'",
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


@requires_docker
def test_container_prpl_mono_mounted(tmp_path: Path) -> None:
    """/robocode/prpl-mono/ is present (bind-mounted from the host submodule)."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(
        config.sandbox_dir,
        "test -d /robocode/prpl-mono && echo mounted || echo absent",
    )
    assert result.returncode == 0, result.stderr
    assert "mounted" in result.stdout


@requires_docker
def test_container_prpl_mono_importable(tmp_path: Path) -> None:
    """prpl-mono packages (relational_structs) are importable via the venv."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(
        config.sandbox_dir,
        f"{DOCKER_PYTHON} -c 'import relational_structs; print(\"OK\")'",
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


@requires_docker
def test_container_robocode_source_not_in_sandbox(tmp_path: Path) -> None:
    """Robocode source is NOT present inside /sandbox/ directly."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(
        config.sandbox_dir,
        "test -d /sandbox/robocode && echo found || echo absent",
    )
    assert "absent" in result.stdout


@requires_docker
def test_container_hook_blocks_write_outside_sandbox(tmp_path: Path) -> None:
    """validate_sandbox.py denies a Write tool call to a path outside /sandbox."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(
        config.sandbox_dir,
        'echo \'{"tool_name": "Write", "tool_input": {"file_path": "/etc/evil"}}\' '
        "| python3 /sandbox/.claude/validate_sandbox.py",
    )
    assert "deny" in result.stdout


@requires_docker
def test_container_hook_allows_write_inside_sandbox(tmp_path: Path) -> None:
    """validate_sandbox.py allows a Write tool call to a path inside /sandbox."""
    config = DockerSandboxConfig(sandbox_dir=tmp_path / "sandbox")
    _setup_sandbox_dir(config)
    result = _run_in_container(
        config.sandbox_dir,
        'echo \'{"tool_name": "Write", "tool_input": {"file_path": "/sandbox/approach.py"}}\' '
        "| python3 /sandbox/.claude/validate_sandbox.py; echo exit=$?",
    )
    assert "exit=0" in result.stdout
    assert "deny" not in result.stdout
